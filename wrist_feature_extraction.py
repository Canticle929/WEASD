#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD Wrist Feature Extraction Script
Extracts time-domain features from windowed wrist data (ACC, BVP, EDA, TEMP).
Handles ACC's triaxial data by computing features per axis (X, Y, Z).
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def extract_time_domain_features(signal_window):
    """
    Extract time-domain features from a 1D signal window.
    
    Parameters:
        signal_window (np.array): 1D array of signal values.
    
    Returns:
        dict: Dictionary of extracted features.
    """
    features = {
        'mean': np.mean(signal_window),
        'std': np.std(signal_window),
        'min': np.min(signal_window),
        'max': np.max(signal_window),
        'range': np.ptp(signal_window),
        'median': np.median(signal_window),
        'q25': np.percentile(signal_window, 25),
        'q75': np.percentile(signal_window, 75),
        'iqr': np.percentile(signal_window, 75) - np.percentile(signal_window, 25),
        'skewness': stats.skew(signal_window),
        'kurtosis': stats.kurtosis(signal_window),
        'rms': np.sqrt(np.mean(np.square(signal_window))),
        'zero_crossings': np.sum(np.diff(np.signbit(signal_window).astype(int)) != 0),
        'diff_mean': np.mean(np.abs(np.diff(signal_window))),
        'diff_std': np.std(np.diff(signal_window)),
        'diff_max': np.max(np.abs(np.diff(signal_window)))
    }
    return features

def process_subject(subject_path, subject_id):
    """
    Process a single subject's windowed data to extract features.
    
    Parameters:
        subject_path (str): Path to the subject's data directory.
        subject_id (str): Identifier for the subject.
    
    Returns:
        pd.DataFrame: Features DataFrame for the subject.
    """
    try:
        # Load windowed data
        acc_windows = np.load(os.path.join(subject_path, 'wrist_ACC_windows.npy'))  # (n, 3500, 3)
        bvp_windows = np.load(os.path.join(subject_path, 'wrist_BVP_windows.npy'))  # (n, 3500)
        eda_windows = np.load(os.path.join(subject_path, 'wrist_EDA_windows.npy'))
        temp_windows = np.load(os.path.join(subject_path, 'wrist_TEMP_windows.npy'))
        labels = np.load(os.path.join(subject_path, 'window_labels.npy'))
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}")
        return pd.DataFrame()

    # Verify window counts
    n_windows = labels.shape[0]
    if not (acc_windows.shape[0] == bvp_windows.shape[0] == eda_windows.shape[0] == temp_windows.shape[0] == n_windows):
        print(f"Window count mismatch for {subject_id}")
        return pd.DataFrame()

    features_list = []
    for i in range(n_windows):
        features = {}

        # Process ACC axes (X, Y, Z)
        acc_win = acc_windows[i]
        for axis_idx, axis in enumerate(['X', 'Y', 'Z']):
            acc_signal = acc_win[:, axis_idx]
            acc_features = extract_time_domain_features(acc_signal)
            for key, val in acc_features.items():
                features[f'ACC_{axis}_{key}'] = val

        # Process BVP
        bvp_signal = bvp_windows[i]
        bvp_features = extract_time_domain_features(bvp_signal)
        for key, val in bvp_features.items():
            features[f'BVP_{key}'] = val

        # Process EDA
        eda_signal = eda_windows[i]
        eda_features = extract_time_domain_features(eda_signal)
        for key, val in eda_features.items():
            features[f'EDA_{key}'] = val

        # Process TEMP
        temp_signal = temp_windows[i]
        temp_features = extract_time_domain_features(temp_signal)
        for key, val in temp_features.items():
            features[f'TEMP_{key}'] = val

        features['label'] = labels[i]
        features['subject_id'] = subject_id
        features_list.append(features)

    return pd.DataFrame(features_list)

def create_feature_dataset(input_dir, output_dir):
    """
    Main function to create feature dataset from windowed wrist data.
    
    Parameters:
        input_dir (str): Directory containing subject subdirectories with windowed data.
        output_dir (str): Directory to save feature dataset and visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # List subject directories (exclude 'merged')
    subjects = [d for d in os.listdir(input_dir) 
                if os.path.isdir(os.path.join(input_dir, d)) and d != 'merged']
    print(f"Found {len(subjects)} subjects")

    all_data = []
    for subject_id in subjects:
        print(f"Processing {subject_id}...")
        subject_path = os.path.join(input_dir, subject_id)
        subject_df = process_subject(subject_path, subject_id)
        if not subject_df.empty:
            all_data.append(subject_df)

    if not all_data:
        print("No data extracted")
        return

    # Combine all subjects' data
    feature_df = pd.concat(all_data, ignore_index=True)
    print(f"Total samples: {len(feature_df)}")

    # Save dataset
    output_path = os.path.join(output_dir, "wrist_feature_dataset.csv")
    feature_df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

    # Generate info file
    with open(os.path.join(output_dir, "feature_info.txt"), 'w') as f:
        f.write(f"Total samples: {len(feature_df)}\n")
        f.write(f"Features: {len(feature_df.columns) - 2}\n")  # Exclude label and subject_id
        f.write("Labels distribution:\n")
        label_counts = feature_df['label'].value_counts()
        for label, count in label_counts.items():
            f.write(f"  Label {label}: {count} ({count/len(feature_df):.2%})\n")

    # Visualizations
    visualize_features(feature_df, output_dir)

def visualize_features(feature_df, output_dir):
    """Generate feature distribution and correlation visualizations."""
    print("Generating visualizations...")
    
    # Boxplots for selected features
    plt.figure(figsize=(18, 12))
    features_to_plot = [
        'ACC_X_mean', 'ACC_Y_std', 'ACC_Z_rms',
        'BVP_mean', 'EDA_std', 'TEMP_median'
    ]
    for i, feat in enumerate(features_to_plot):
        plt.subplot(2, 3, i+1)
        feature_df.boxplot(column=feat, by='label', grid=False)
        plt.title(feat)
    plt.suptitle('Feature Distributions by Label')
    plt.savefig(os.path.join(output_dir, 'feature_boxplots.png'))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(20, 18))
    numeric_df = feature_df.select_dtypes(include=[np.number]).drop(['label'], axis=1)
    corr_matrix = numeric_df.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'))
    plt.close()

if __name__ == "__main__":
    input_dir = "/kaggle/working/WESAD/windowed_wrist_data"  # Update path
    output_dir = "kaggle/working/WESAD/wrist_features"     # Update path

    print("="*80)
    print("WESAD Wrist Feature Extraction")
    print("="*80)
    create_feature_dataset(input_dir, output_dir)
    print("\nFeature extraction completed!")
    print("="*80)