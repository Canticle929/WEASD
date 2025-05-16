#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD Data Cleaning Script
Specifically designed to extract ECG, EDA, and Resp signals from WESAD raw data,
while only retaining data with labels 1 (baseline) and 2 (stress)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

def load_pickle_file(file_path):
    """
    Load pickle file from WESAD dataset
    
    Args:
        file_path: Path to the pickle file
    
    Returns:
        data: Dictionary containing the data
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def clean_wesad_data(data, target_labels=[1, 2]):
    """
    Clean WESAD data by retaining only specified labels
    
    Args:
        data: WESAD data dictionary
        target_labels: List of labels to retain, default [1, 2] (baseline and stress)
    
    Returns:
        clean_data: Cleaned data dictionary containing:
            - chest_ECG: Chest ECG signal
            - chest_EDA: Chest EDA signal
            - chest_Resp: Chest respiratory signal
            - labels: Corresponding labels
            - label_names: Label meanings
    """
    # Extract chest signals
    chest_data = data['signal']['chest']
    
    # Extract required signals
    chest_ECG = chest_data['ECG'].flatten()
    chest_EDA = chest_data['EDA'].flatten()
    chest_Resp = chest_data['Resp'].flatten()
    
    # Extract labels
    labels = data['label'].flatten()
    
    # Find indices of target labels
    target_indices = np.isin(labels, target_labels)
    
    # Check if target labels exist
    if not np.any(target_indices):
        print(f"Warning: No data points found for labels {target_labels}")
        
        # Show distribution of all labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Label {label}: {count} samples")
        
        # Return empty data
        return {
            'chest_ECG': np.array([]),
            'chest_EDA': np.array([]),
            'chest_Resp': np.array([]),
            'labels': np.array([]),
            'label_names': {1: 'baseline', 2: 'stress'},
            'sampling_rate': {'chest': 700}
        }
    
    # Extract data corresponding to target labels
    clean_ECG = chest_ECG[target_indices]
    clean_EDA = chest_EDA[target_indices]
    clean_Resp = chest_Resp[target_indices]
    clean_labels = labels[target_indices]
    
    # Create label name dictionary
    label_names = {
        1: 'baseline',  # Baseline state
        2: 'stress'     # Stress state
    }
    
    # Check size of extracted data
    print(f"Original data points: {len(labels)}")
    print(f"Cleaned data points: {len(clean_labels)}")
    
    # Count samples for each label
    for label in target_labels:
        count = np.sum(clean_labels == label)
        print(f"Label {label} ({label_names[label]}): {count} samples")
    
    # Define the known sampling rate for chest signals
    chest_sampling_rate = 700 # Hz, according to WESAD readme for RespiBAN

    return {
        'chest_ECG': clean_ECG,
        'chest_EDA': clean_EDA,
        'chest_Resp': clean_Resp,
        'labels': clean_labels,
        'label_names': label_names,
        'sampling_rate': {'chest': chest_sampling_rate}
    }

def process_subject(subject_path, output_dir, target_labels=[1, 2]):
    """
    Process data for a single subject
    
    Args:
        subject_path: Path to subject data file
        output_dir: Output directory
        target_labels: List of labels to retain, default [1, 2] (baseline and stress)
    
    Returns:
        success: Whether processing was successful
    """
    # Extract subject ID
    subject_id = os.path.basename(subject_path).split('.')[0]
    print(f"Processing subject {subject_id}...")
    
    # Load data
    data = load_pickle_file(subject_path)
    
    # Clean data
    clean_data = clean_wesad_data(data, target_labels)
    
    # Skip saving if no target label data exists
    if len(clean_data['labels']) == 0:
        print(f"Subject {subject_id} has no data points for labels {target_labels}, skipping")
        return False
    
    # Create save directory
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # Save cleaned data
    np.save(os.path.join(subject_output_dir, 'chest_ECG.npy'), clean_data['chest_ECG'])
    np.save(os.path.join(subject_output_dir, 'chest_EDA.npy'), clean_data['chest_EDA'])
    np.save(os.path.join(subject_output_dir, 'chest_Resp.npy'), clean_data['chest_Resp'])
    np.save(os.path.join(subject_output_dir, 'labels.npy'), clean_data['labels'])
    
    # Save sampling rate and label information
    with open(os.path.join(subject_output_dir, 'info.txt'), 'w') as f:
        f.write(f"Data information for subject {subject_id}\n")
        f.write("=========================\n\n")
        f.write(f"Sampling rate: {clean_data['sampling_rate']['chest']} Hz\n\n")
        f.write("Label information:\n")
        for label, name in clean_data['label_names'].items():
            count = np.sum(clean_data['labels'] == label)
            f.write(f"  Label {label} ({name}): {count} samples\n")
        
        # Calculate duration for each label
        sampling_rate = clean_data['sampling_rate']['chest']
        for label, name in clean_data['label_names'].items():
            count = np.sum(clean_data['labels'] == label)
            duration_seconds = count / sampling_rate
            duration_minutes = duration_seconds / 60
            f.write(f"  Label {label} ({name}) duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)\n")
    
    # Create simple visualization for each signal
    plt.figure(figsize=(15, 10))
    
    # ECG signal
    plt.subplot(3, 1, 1)
    for label in target_labels:
        indices = clean_data['labels'] == label
        if np.any(indices):
            # Plot only first 10000 points to avoid large plots
            max_points = min(10000, np.sum(indices))
            idx = np.where(indices)[0][:max_points]
            plt.plot(idx, clean_data['chest_ECG'][idx], 
                    label=f"Label {label} ({clean_data['label_names'][label]})")
    plt.title(f"Subject {subject_id} - ECG Signal")
    plt.legend()
    plt.grid(True)
    
    # EDA signal
    plt.subplot(3, 1, 2)
    for label in target_labels:
        indices = clean_data['labels'] == label
        if np.any(indices):
            max_points = min(10000, np.sum(indices))
            idx = np.where(indices)[0][:max_points]
            plt.plot(idx, clean_data['chest_EDA'][idx], 
                    label=f"Label {label} ({clean_data['label_names'][label]})")
    plt.title(f"Subject {subject_id} - EDA Signal")
    plt.legend()
    plt.grid(True)
    
    # Resp signal
    plt.subplot(3, 1, 3)
    for label in target_labels:
        indices = clean_data['labels'] == label
        if np.any(indices):
            max_points = min(10000, np.sum(indices))
            idx = np.where(indices)[0][:max_points]
            plt.plot(idx, clean_data['chest_Resp'][idx], 
                    label=f"Label {label} ({clean_data['label_names'][label]})")
    plt.title(f"Subject {subject_id} - Resp Signal")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(subject_output_dir, 'signal_visualization.png'))
    plt.close()
    
    return True

def process_wesad_dataset(data_dir, output_dir, target_labels=[1, 2]):
    """
    Process entire WESAD dataset
    
    Args:
        data_dir: Path to original data directory
        output_dir: Path to output directory
        target_labels: List of labels to retain, default [1, 2] (baseline and stress)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all pickle files
    subject_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.pkl'):
                subject_files.append(os.path.join(root, file))
    
    print(f"Found {len(subject_files)} subject data files")
    
    # Process data for each subject
    successful_subjects = 0
    for subject_file in subject_files:
        success = process_subject(subject_file, output_dir, target_labels)
        if success:
            successful_subjects += 1
    
    print(f"Successfully processed {successful_subjects}/{len(subject_files)} subjects")
    
    # Create dataset summary information
    create_dataset_summary(output_dir)

def create_dataset_summary(output_dir):
    """
    Create dataset summary information
    
    Args:
        output_dir: Output directory for cleaned data
    """
    # Get all subject directories
    subject_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('S')]
    
    if not subject_dirs:
        print("No valid subject data found")
        return
    
    # Statistics
    total_baseline_samples = 0
    total_stress_samples = 0
    baseline_durations = []
    stress_durations = []
    
    # Create statistics table data
    stats_data = []
    
    for subject in sorted(subject_dirs):
        subject_dir = os.path.join(output_dir, subject)
        labels_path = os.path.join(subject_dir, 'labels.npy')
        
        if not os.path.exists(labels_path):
            continue
        
        # Load labels
        labels = np.load(labels_path)
        
        # Count samples for each label
        baseline_count = np.sum(labels == 1)
        stress_count = np.sum(labels == 2)
        
        # Load data from first subject to get sampling rate
        if len(stats_data) == 0:
            info_path = os.path.join(subject_dir, 'info.txt')
            with open(info_path, 'r') as f:
                info_text = f.read()
                for line in info_text.split('\n'):
                    if line.startswith('Sampling rate:'):
                        sampling_rate = float(line.split(':')[1].strip().split(' ')[0])
                        break
        
        # Calculate duration (seconds)
        baseline_duration = baseline_count / sampling_rate
        stress_duration = stress_count / sampling_rate
        
        # Add to statistics data
        stats_data.append({
            'Subject': subject,
            'Baseline Samples': baseline_count,
            'Stress Samples': stress_count,
            'Baseline Duration (seconds)': baseline_duration,
            'Stress Duration (seconds)': stress_duration,
            'Baseline Duration (minutes)': baseline_duration / 60,
            'Stress Duration (minutes)': stress_duration / 60
        })
        
        # Update totals
        total_baseline_samples += baseline_count
        total_stress_samples += stress_count
        
        if baseline_count > 0:
            baseline_durations.append(baseline_duration)
        if stress_count > 0:
            stress_durations.append(stress_duration)
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(stats_data)
    csv_path = os.path.join(output_dir, 'dataset_stats.csv')
    df.to_csv(csv_path, index=False)
    
    # Calculate statistics summary
    df_summary = pd.DataFrame({
        'Category': ['Baseline', 'Stress'],
        'Total Samples': [total_baseline_samples, total_stress_samples],
        'Total Duration (seconds)': [sum(baseline_durations), sum(stress_durations)],
        'Total Duration (minutes)': [sum(baseline_durations) / 60, sum(stress_durations) / 60],
        'Average Duration (seconds/subject)': [
            sum(baseline_durations) / len([d for d in baseline_durations if d > 0]) if baseline_durations else 0,
            sum(stress_durations) / len([d for d in stress_durations if d > 0]) if stress_durations else 0
        ],
        'Shortest Duration (seconds)': [
            min(baseline_durations) if baseline_durations else 0,
            min(stress_durations) if stress_durations else 0
        ],
        'Longest Duration (seconds)': [
            max(baseline_durations) if baseline_durations else 0,
            max(stress_durations) if stress_durations else 0
        ]
    })
    
    summary_csv_path = os.path.join(output_dir, 'dataset_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    
    # Create summary report
    with open(os.path.join(output_dir, 'dataset_report.txt'), 'w') as f:
        f.write("WESAD Dataset Cleaning Report\n")
        f.write("===================\n\n")
        f.write(f"Contains {len(subject_dirs)} valid subjects\n\n")
        
        f.write("Label statistics:\n")
        f.write(f"  Baseline (label 1): Total {total_baseline_samples} samples, "
                f"total duration {sum(baseline_durations):.2f} seconds ({sum(baseline_durations)/60:.2f} minutes)\n")
        f.write(f"  Stress (label 2): Total {total_stress_samples} samples, "
                f"total duration {sum(stress_durations):.2f} seconds ({sum(stress_durations)/60:.2f} minutes)\n\n")
        
        f.write("Dataset balance:\n")
        total_samples = total_baseline_samples + total_stress_samples
        if total_samples > 0:
            baseline_percent = (total_baseline_samples / total_samples) * 100
            stress_percent = (total_stress_samples / total_samples) * 100
            f.write(f"  Baseline sample percentage: {baseline_percent:.2f}%\n")
            f.write(f"  Stress sample percentage: {stress_percent:.2f}%\n\n")
        
        f.write("Detailed subject information saved to dataset_stats.csv\n")
        f.write("Dataset summary statistics saved to dataset_summary.csv\n")
    
    print(f"Dataset summary report created, saved to {output_dir}")
    
    # Create label distribution visualization
    plt.figure(figsize=(10, 6))
    
    # Samples count bar chart
    plt.subplot(1, 2, 1)
    plt.bar(['Baseline (label 1)', 'Stress (label 2)'], [total_baseline_samples, total_stress_samples])
    plt.title('Samples count distribution')
    plt.ylabel('Samples count')
    plt.grid(True, axis='y')
    
    # Duration bar chart
    plt.subplot(1, 2, 2)
    plt.bar(['Baseline (label 1)', 'Stress (label 2)'], [sum(baseline_durations)/60, sum(stress_durations)/60])
    plt.title('Duration distribution')
    plt.ylabel('Total duration (minutes)')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'label_distribution.png'), dpi=300)
    plt.close()
    
    # Create label distribution visualization for each subject
    plt.figure(figsize=(12, 8))
    
    subjects = [d['Subject'] for d in stats_data]
    baseline_minutes = [d['Baseline Duration (minutes)'] for d in stats_data]
    stress_minutes = [d['Stress Duration (minutes)'] for d in stats_data]
    
    x = np.arange(len(subjects))
    width = 0.35
    
    plt.bar(x - width/2, baseline_minutes, width, label='Baseline (label 1)')
    plt.bar(x + width/2, stress_minutes, width, label='Stress (label 2)')
    
    plt.xlabel('Subject')
    plt.ylabel('Duration (minutes)')
    plt.title('Label distribution for each subject')
    plt.xticks(x, subjects)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_distribution.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    # Configure parameters
    data_dir = "/Volumes/xcy/TeamProject/WESAD"  # WESAD original data directory (parent of S* folders)
    output_dir = "/Volumes/xcy/TeamProject/WESAD/cleaned_data"  # Output directory for cleaned data
    target_labels = [1, 2]  # Labels to retain: 1=baseline, 2=stress
    
    print("="*80)
    print("WESAD Data Cleaning")
    print("="*80)
    
    # Process dataset
    process_wesad_dataset(data_dir, output_dir, target_labels)
    
    print("\nData cleaning completed!")
    print(f"Cleaned data saved in {output_dir}")
    print("="*80) 