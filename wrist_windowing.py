#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD Wrist Data Windowing Script
Processes the upsampled 700Hz wrist data (ACC, BVP, EDA, TEMP),
using 5-second windows with 2.5-second steps (50% overlap), 
and determines window labels using the mode of individual sample labels.
"""

import os
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt

def window_wrist_data(input_dir, output_dir, window_size=5, overlap=0.5):
    """
    Applies windowing to the upsampled 700Hz wrist data
    
    Args:
        input_dir: Input directory path (with upsampled 700Hz wrist data)
        output_dir: Output directory path for windowed data
        window_size: Window size in seconds, default 5 seconds
        overlap: Window overlap ratio, default 0.5 (50%)
    """
    # Calculate constants
    sampling_rate = 700  # All signals are upsampled to 700Hz
    window_samples = int(window_size * sampling_rate)  # Number of samples per window
    step_samples = int(window_samples * (1 - overlap))  # Window step size (in samples)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all subject directories (excluding 'merged' directory)
    subject_dirs = []
    for entry in os.listdir(input_dir):
        entry_path = os.path.join(input_dir, entry)
        if os.path.isdir(entry_path) and entry != 'merged':
            subject_dirs.append(entry_path)
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    # Lists to collect all windows
    all_acc_windows = []
    all_bvp_windows = []
    all_eda_windows = []
    all_temp_windows = []
    all_window_labels = []
    
    # Process each subject
    window_counts = {}  # Track window count by label
    
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        print(f"Processing subject {subject_id}...")
        
        # Load 700Hz signal data
        try:
            # Check if files exist
            acc_path = os.path.join(subject_dir, 'wrist_ACC_700Hz.npy')
            bvp_path = os.path.join(subject_dir, 'wrist_BVP_700Hz.npy')
            eda_path = os.path.join(subject_dir, 'wrist_EDA_700Hz.npy')
            temp_path = os.path.join(subject_dir, 'wrist_TEMP_700Hz.npy')
            labels_path = os.path.join(subject_dir, 'labels_700Hz.npy')
            
            if not all(os.path.exists(p) for p in [acc_path, bvp_path, eda_path, temp_path, labels_path]):
                print(f"   Subject {subject_id} has incomplete data files, skipping")
                continue
                
            # Load data
            acc = np.load(acc_path)  # Shape (N, 3)
            bvp = np.load(bvp_path)  # Shape (N,)
            eda = np.load(eda_path)  # Shape (N,)
            temp = np.load(temp_path)  # Shape (N,)
            labels = np.load(labels_path)  # Shape (N,)
        except Exception as e:
            print(f"   Error loading data for subject {subject_id}: {e}")
            continue
        
        # Verify all signals have consistent lengths
        signal_len = len(labels)
        if not all(len(s) == signal_len for s in [bvp, eda, temp]) or len(acc) != signal_len:
            print(f"   Subject {subject_id} has inconsistent signal lengths, skipping")
            print(f"   Labels: {signal_len}, ACC: {len(acc)}, BVP: {len(bvp)}, EDA: {len(eda)}, TEMP: {len(temp)}")
            continue
            
        # Display signal length and label statistics
        baseline_count = np.sum(labels == 1)
        stress_count = np.sum(labels == 2)
        print(f"   Data length: {signal_len} samples ({signal_len/sampling_rate:.1f} seconds)")
        print(f"   Baseline samples: {baseline_count} ({baseline_count/sampling_rate:.1f} seconds)")
        print(f"   Stress samples: {stress_count} ({stress_count/sampling_rate:.1f} seconds)")
        
        # Determine window count
        num_windows = (signal_len - window_samples) // step_samples + 1
        print(f"   Will create {num_windows} windows (each {window_size} seconds, step {window_size * (1-overlap)} seconds)")
        
        # Create subject output directory
        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Create window arrays for this subject
        acc_windows = []
        bvp_windows = []
        eda_windows = []
        temp_windows = []
        window_labels = []
        
        for i in range(num_windows):
            # Calculate window start and end indices
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            
            # Ensure we have enough samples
            if end_idx > signal_len:
                break
                
            # Extract window data
            acc_win = acc[start_idx:end_idx]
            bvp_win = bvp[start_idx:end_idx]
            eda_win = eda[start_idx:end_idx]
            temp_win = temp[start_idx:end_idx]
            labels_win = labels[start_idx:end_idx]
            
            # Calculate window label (using mode)
            # Note: If multiple values have the same frequency, mode returns the smallest
            # For binary labels (1,2), this means if frequencies are equal, it returns 1 (baseline)
            win_label = stats.mode(labels_win, keepdims=False)[0]
            
            # Add window to lists
            acc_windows.append(acc_win)
            bvp_windows.append(bvp_win)
            eda_windows.append(eda_win)
            temp_windows.append(temp_win)
            window_labels.append(win_label)
            
            # Update window count statistics
            if win_label in window_counts:
                window_counts[win_label] += 1
            else:
                window_counts[win_label] = 1
        
        # Convert to numpy arrays
        acc_windows = np.array(acc_windows)
        bvp_windows = np.array(bvp_windows)
        eda_windows = np.array(eda_windows)
        temp_windows = np.array(temp_windows)
        window_labels = np.array(window_labels)
        
        # Save subject's windowed data
        np.save(os.path.join(subject_output_dir, 'wrist_ACC_windows.npy'), acc_windows)
        np.save(os.path.join(subject_output_dir, 'wrist_BVP_windows.npy'), bvp_windows)
        np.save(os.path.join(subject_output_dir, 'wrist_EDA_windows.npy'), eda_windows)
        np.save(os.path.join(subject_output_dir, 'wrist_TEMP_windows.npy'), temp_windows)
        np.save(os.path.join(subject_output_dir, 'window_labels.npy'), window_labels)
        
        # Record statistics
        with open(os.path.join(subject_output_dir, 'info.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Window size: {window_size} seconds ({window_samples} samples)\n")
            f.write(f"Window step: {window_size * (1-overlap)} seconds ({step_samples} samples)\n")
            f.write(f"Window overlap: {overlap * 100}%\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n")
            f.write(f"Total windows: {len(window_labels)}\n")
            f.write(f"Baseline windows: {np.sum(window_labels == 1)}\n")
            f.write(f"Stress windows: {np.sum(window_labels == 2)}\n")
            f.write(f"ACC windows shape: {acc_windows.shape}\n")
            f.write(f"BVP windows shape: {bvp_windows.shape}\n")
            f.write(f"EDA windows shape: {eda_windows.shape}\n")
            f.write(f"TEMP windows shape: {temp_windows.shape}\n")
        
        # Create label distribution visualization
        plt.figure(figsize=(10, 6))
        label_counts = [np.sum(window_labels == 1), np.sum(window_labels == 2)]
        plt.bar(['Baseline (1)', 'Stress (2)'], label_counts, color=['green', 'red'])
        plt.title(f"Subject {subject_id} Window Label Distribution")
        plt.ylabel("Window Count")
        plt.savefig(os.path.join(subject_output_dir, 'label_distribution.png'))
        plt.close()
        
        print(f"   Saved {len(window_labels)} windows (Baseline: {np.sum(window_labels == 1)}, Stress: {np.sum(window_labels == 2)})")
        
        # Add to global lists
        all_acc_windows.extend(acc_windows)
        all_bvp_windows.extend(bvp_windows)
        all_eda_windows.extend(eda_windows)
        all_temp_windows.extend(temp_windows)
        all_window_labels.extend(window_labels)
    
    # Merge all subjects' data (if any)
    if all_window_labels:
        print(f"\nMerging window data from all subjects...")
        
        # Convert to numpy arrays
        all_acc_windows = np.array(all_acc_windows)
        all_bvp_windows = np.array(all_bvp_windows)
        all_eda_windows = np.array(all_eda_windows)
        all_temp_windows = np.array(all_temp_windows)
        all_window_labels = np.array(all_window_labels)
        
        # Create merged directory
        merged_dir = os.path.join(output_dir, 'merged')
        os.makedirs(merged_dir, exist_ok=True)
        
        # Save merged window data
        np.save(os.path.join(merged_dir, 'wrist_ACC_windows.npy'), all_acc_windows)
        np.save(os.path.join(merged_dir, 'wrist_BVP_windows.npy'), all_bvp_windows)
        np.save(os.path.join(merged_dir, 'wrist_EDA_windows.npy'), all_eda_windows)
        np.save(os.path.join(merged_dir, 'wrist_TEMP_windows.npy'), all_temp_windows)
        np.save(os.path.join(merged_dir, 'window_labels.npy'), all_window_labels)
        
        # Record merged statistics
        with open(os.path.join(merged_dir, 'info.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Merged window data from {len(subject_dirs)} subjects\n")
            f.write(f"Window size: {window_size} seconds ({window_samples} samples)\n")
            f.write(f"Window step: {window_size * (1-overlap)} seconds ({step_samples} samples)\n")
            f.write(f"Window overlap: {overlap * 100}%\n")
            f.write(f"Sampling rate: {sampling_rate} Hz\n")
            f.write(f"Total windows: {len(all_window_labels)}\n")
            f.write(f"Baseline windows: {np.sum(all_window_labels == 1)}\n")
            f.write(f"Stress windows: {np.sum(all_window_labels == 2)}\n")
            f.write(f"ACC windows shape: {all_acc_windows.shape}\n")
            f.write(f"BVP windows shape: {all_bvp_windows.shape}\n")
            f.write(f"EDA windows shape: {all_eda_windows.shape}\n")
            f.write(f"TEMP windows shape: {all_temp_windows.shape}\n")
        
        # Create global label distribution visualization
        plt.figure(figsize=(10, 6))
        all_label_counts = [np.sum(all_window_labels == 1), np.sum(all_window_labels == 2)]
        plt.bar(['Baseline (1)', 'Stress (2)'], all_label_counts, color=['green', 'red'])
        plt.title("All Subjects Window Label Distribution")
        plt.ylabel("Window Count")
        plt.savefig(os.path.join(merged_dir, 'label_distribution.png'))
        plt.close()
        
        print(f"Merged data shapes:")
        print(f"  ACC windows: {all_acc_windows.shape}")
        print(f"  BVP windows: {all_bvp_windows.shape}")
        print(f"  EDA windows: {all_eda_windows.shape}")
        print(f"  TEMP windows: {all_temp_windows.shape}")
        print(f"  Labels: {all_window_labels.shape}")
        print(f"  Baseline windows: {np.sum(all_window_labels == 1)}")
        print(f"  Stress windows: {np.sum(all_window_labels == 2)}")
    else:
        print("No valid data to merge")

if __name__ == "__main__":
    # Configure parameters
    input_dir = "/Volumes/xcy/TeamProject/WESAD/cleaned_wrist_data_700Hz"  # Directory with upsampled 700Hz wrist data
    output_dir = "/Volumes/xcy/TeamProject/WESAD/windowed_wrist_data"  # Output directory for windowed data
    window_size = 5  # Window size in seconds
    overlap = 0.5  # Window overlap ratio
    
    print("="*80)
    print("WESAD Wrist Data Windowing")
    print("="*80)
    
    window_wrist_data(input_dir, output_dir, window_size, overlap)
    
    print("\nWindowing process completed!")
    print(f"Windowed data saved in {output_dir}")
    print("="*80) 