#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD Feature Extraction Script
Extracts statistical features from windowed signals for machine learning model training.
Loads data from the merged windowed dataset and calculates time-domain features.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def extract_time_domain_features(window):
    """
    Extract time-domain features from a signal window
    
    Parameters:
        window: A single window of signal data (numpy array)
    
    Returns:
        features: Dictionary of extracted features
    """
    features = {
        # Basic statistics
        'mean': np.mean(window),
        'std': np.std(window),
        'min': np.min(window),
        'max': np.max(window),
        'range': np.max(window) - np.min(window),
        'median': np.median(window),
        
        # Quartiles
        'q25': np.percentile(window, 25),
        'q75': np.percentile(window, 75),
        'iqr': np.percentile(window, 75) - np.percentile(window, 25),
        
        # Shape statistics
        'skewness': stats.skew(window),
        'kurtosis': stats.kurtosis(window),
        
        # Variability
        'rms': np.sqrt(np.mean(np.square(window))),
        'zero_crossings': np.sum(np.diff(np.signbit(window).astype(int)) != 0),
        
        # First-order difference statistics (rate of change)
        'diff_mean': np.mean(np.abs(np.diff(window))),
        'diff_std': np.std(np.diff(window)),
        'diff_max': np.max(np.abs(np.diff(window)))
    }
    
    return features

def extract_features_from_windows(signal_windows, signal_name):
    """
    Extract features from all windows of a signal
    
    Parameters:
        signal_windows: Array of signal windows (n_windows, window_length)
        signal_name: Name of the signal (for feature naming)
    
    Returns:
        features_df: DataFrame with extracted features for all windows
    """
    print(f"Extracting features from {signal_name} windows...")
    
    n_windows = signal_windows.shape[0]
    features_list = []
    
    for i in range(n_windows):
        window = signal_windows[i]
        features = extract_time_domain_features(window)
        features_list.append(features)
    
    # Create DataFrame with features
    features_df = pd.DataFrame(features_list)
    
    # Rename columns to include signal name
    features_df = features_df.add_prefix(f"{signal_name}_")
    
    return features_df

def create_feature_dataset(input_dir, output_dir):
    """
    Create a feature dataset from the merged windowed data
    
    Parameters:
        input_dir: Directory with merged windowed data
        output_dir: Output directory for the feature dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load window labels and subject IDs
    try:
        labels = np.load(os.path.join(input_dir, "window_labels.npy"))
        subject_ids = np.load(os.path.join(input_dir, "subject_ids.npy"))
    except Exception as e:
        print(f"Error loading labels or subject IDs: {e}")
        return
    
    print(f"Found {len(labels)} windows with labels")
    
    # Count label distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  Label {label}: {count} windows ({count/len(labels)*100:.2f}%)")
    
    # Load all signal windows
    signal_dfs = []
    
    # List all window files
    window_files = [f for f in os.listdir(input_dir) if f.endswith("_windows.npy")]
    
    for window_file in window_files:
        signal_name = window_file.replace("_windows.npy", "")
        
        try:
            signal_windows = np.load(os.path.join(input_dir, window_file))
            print(f"Loaded {signal_name} windows: {signal_windows.shape}")
            
            # Extract features from this signal's windows
            signal_features_df = extract_features_from_windows(signal_windows, signal_name)
            signal_dfs.append(signal_features_df)
            
        except Exception as e:
            print(f"Error processing {signal_name}: {e}")
    
    if not signal_dfs:
        print("No valid signal features extracted")
        return
    
    # Combine all signal features
    combined_features = pd.concat(signal_dfs, axis=1)
    
    # Add labels and subject IDs
    combined_features['label'] = labels
    combined_features['subject_id'] = subject_ids
    
    # Save feature dataset to CSV
    feature_file = os.path.join(output_dir, "feature_dataset.csv")
    combined_features.to_csv(feature_file, index=False)
    
    # Save feature dataset information
    with open(os.path.join(output_dir, "feature_info.txt"), "w") as f:
        f.write("WESAD Feature Dataset Information\n")
        f.write("===============================\n\n")
        f.write(f"Total samples: {len(combined_features)}\n")
        f.write(f"Number of features: {len(combined_features.columns) - 2}\n\n")  # Subtract 'label' and 'subject_id'
        
        f.write("Label distribution:\n")
        for label, count in zip(unique_labels, counts):
            label_name = "Baseline" if label == 1 else "Stress" if label == 2 else f"Unknown ({label})"
            f.write(f"  Label {label} ({label_name}): {count} samples ({count/len(labels)*100:.2f}%)\n")
        
        f.write("\nFeature list:\n")
        for col in combined_features.columns:
            if col not in ['label', 'subject_id']:
                f.write(f"  {col}\n")
    
    print(f"\nFeature extraction completed!")
    print(f"Extracted {len(combined_features.columns) - 2} features from {len(combined_features)} windows")
    print(f"Feature dataset saved to {feature_file}")

def visualize_features(feature_file, output_dir):
    """
    Create basic visualizations of the extracted features
    
    Parameters:
        feature_file: Path to the feature dataset CSV file
        output_dir: Output directory for visualizations
    """
    # Load feature dataset
    df = pd.read_csv(feature_file)
    
    print("Creating feature visualizations...")
    
    # 1. Feature boxplots grouped by label
    plt.figure(figsize=(15, 10))
    
    # Select a subset of representative features for visualization
    signal_types = ['chest_ECG', 'chest_EDA', 'chest_Resp']
    feature_types = ['mean', 'std', 'range']
    
    for i, signal in enumerate(signal_types):
        for j, feature in enumerate(feature_types):
            feature_name = f"{signal}_{feature}"
            if feature_name in df.columns:
                plt.subplot(3, 3, i*3 + j + 1)
                df.boxplot(column=feature_name, by='label')
                plt.title(f"{signal} {feature}")
                plt.suptitle("")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_boxplots.png"))
    plt.close()
    
    # 2. Correlation heatmap of features
    plt.figure(figsize=(14, 12))
    
    # Select features only (exclude label and subject_id)
    features_only = df.drop(['label', 'subject_id'], axis=1)
    
    # Calculate and plot correlation matrix
    corr_matrix = features_only.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Feature Correlation Heatmap")
    
    # Due to the large number of features, skip tick labels
    plt.tick_params(axis='both', which='both', labelsize=6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation.png"))
    plt.close()
    
    print("Visualizations created and saved to output directory")

if __name__ == "__main__":
    # Configure parameters
    input_dir = "/Volumes/xcy/TeamProject/WESAD/windowed_data/merged"  # Merged windowed data directory
    output_dir = "/Volumes/xcy/TeamProject/WESAD/features"  # Feature dataset output directory
    
    print("="*80)
    print("WESAD Feature Extraction")
    print("="*80)
    
    # Extract features
    create_feature_dataset(input_dir, output_dir)
    
    # Create visualizations
    feature_file = os.path.join(output_dir, "feature_dataset.csv")
    if os.path.exists(feature_file):
        visualize_features(feature_file, output_dir)
    
    print("\nFeature extraction and visualization completed!")
    print(f"Feature dataset saved in {output_dir}")
    print("="*80) 