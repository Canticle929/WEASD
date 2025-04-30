#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD Wrist Data Extraction Script
Extracts ACC, BVP, EDA, and TEMP signals from the wrist device, 
upsamples them to 700Hz to match label frequency,
and retains only data with labels 1 (baseline) and 2 (stress).
"""

import os
import pickle
import numpy as np
# Import the resample function
from scipy.signal import resample 

def extract_wesad_wrist_data(data_dir, output_dir):
    """
    Extracts, upsamples (to 700Hz), and cleans WESAD wrist data (ACC, BVP, EDA, TEMP), 
    retaining only samples with labels 1 or 2.
    
    Args:
        data_dir (str): Path to the WESAD data directory.
        output_dir (str): Path to the directory where cleaned wrist data will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    subject_dirs = []
    for entry in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith('S') and not entry.startswith('S_'):
            subject_dirs.append(entry_path)
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    # Data collection lists for merging
    all_acc = []
    all_bvp = []
    all_eda = []
    all_temp = []
    all_labels = []
    
    # Define original wrist sampling rates (Hz) - For reference
    # wrist_sampling_rates = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}
    target_sampling_rate = 700 # Target sampling rate is 700 Hz (matching labels)

    processed_subjects = 0

    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        pickle_file = os.path.join(subject_dir, f"{subject_id}.pkl")
        
        if not os.path.exists(pickle_file):
            print(f"Subject {subject_id} data file {pickle_file} does not exist, skipping")
            continue
            
        print(f"Processing subject {subject_id}...")
        
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"Failed to load subject {subject_id} data: {e}")
            continue
        
        # Check if wrist data exists
        if 'wrist' not in data['signal']:
             print(f"   Subject {subject_id} has no wrist data, skipping")
             continue

        wrist_data = data['signal']['wrist']
        labels = data['label'].flatten()
        target_len = len(labels) # Target length based on 700Hz labels

        # Extract signals, handle potential missing keys
        try:
            acc_orig = wrist_data['ACC'] # Shape (N, 3)
            bvp_orig = wrist_data['BVP'].flatten()
            eda_orig = wrist_data['EDA'].flatten()
            temp_orig = wrist_data['TEMP'].flatten()
        except KeyError as e:
            print(f"   Subject {subject_id} missing wrist modality: {e}, skipping subject.")
            continue

        # Upsample signals to target length (700 Hz)
        print(f"   Upsampling wrist signals to {target_len} samples (700 Hz)...")
        try:
            # Resample ACC along the time axis (axis=0)
            acc_resampled = resample(acc_orig, target_len, axis=0)
            bvp_resampled = resample(bvp_orig, target_len)
            eda_resampled = resample(eda_orig, target_len)
            temp_resampled = resample(temp_orig, target_len)
        except Exception as e:
             print(f"   Subject {subject_id}: Error during resampling: {e}. Skipping subject.")
             continue

        # Create mask for labels 1 and 2
        mask = (labels == 1) | (labels == 2)
        
        if not np.any(mask):
            print(f"   Subject {subject_id} has no samples with labels 1 or 2, skipping")
            continue
        
        # Apply mask to the RESAMPLED signals and labels
        try:
            # Ensure mask length matches resampled signal length (should always match label length)
            if len(mask) != target_len:
                 print(f"   Subject {subject_id}: Mask length ({len(mask)}) mismatch with target length ({target_len}). Skipping.")
                 continue

            acc_clean = acc_resampled[mask]
            bvp_clean = bvp_resampled[mask]
            eda_clean = eda_resampled[mask]
            temp_clean = temp_resampled[mask]
            labels_clean = labels[mask]

        except IndexError as e:
            # This error shouldn't happen now if resampling is correct, but keep for safety
            print(f"   Subject {subject_id}: Error applying mask after resampling ({e}). Skipping subject.")
            print(f"   Label length: {len(labels)}, Mask length: {len(mask)}, Target length: {target_len}")
            print(f"   Resampled ACC shape: {acc_resampled.shape}, BVP length: {len(bvp_resampled)}")
            continue 

        baseline_count = np.sum(labels_clean == 1)
        stress_count = np.sum(labels_clean == 2)
        print(f"   Baseline samples (at 700Hz): {baseline_count}, Stress samples (at 700Hz): {stress_count}")
        
        # Save cleaned (and upsampled to 700Hz) data for this subject
        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        np.save(os.path.join(subject_output_dir, 'wrist_ACC_700Hz.npy'), acc_clean)
        np.save(os.path.join(subject_output_dir, 'wrist_BVP_700Hz.npy'), bvp_clean)
        np.save(os.path.join(subject_output_dir, 'wrist_EDA_700Hz.npy'), eda_clean)
        np.save(os.path.join(subject_output_dir, 'wrist_TEMP_700Hz.npy'), temp_clean)
        np.save(os.path.join(subject_output_dir, 'labels_700Hz.npy'), labels_clean) # Also save labels for clarity
        
        # Record sampling rates and stats
        with open(os.path.join(subject_output_dir, 'info.txt'), 'w') as f:
            # Original rates for reference, but saved data is 700Hz
            # f.write(f"Original Wrist Sampling Rates (Hz): {wrist_sampling_rates}\n") 
            f.write(f"Saved Data Sampling Rate (Hz): {target_sampling_rate}\n") 
            f.write(f"Baseline samples (at {target_sampling_rate}Hz): {baseline_count}\n")
            f.write(f"Stress samples (at {target_sampling_rate}Hz): {stress_count}\n")
            f.write(f"Total selected samples (at {target_sampling_rate}Hz): {len(labels_clean)}\n")
        
        # Add to global lists
        all_acc.append(acc_clean)
        all_bvp.append(bvp_clean)
        all_eda.append(eda_clean)
        all_temp.append(temp_clean)
        all_labels.append(labels_clean)
        processed_subjects += 1
    
    # Merge data if any subjects were processed
    if all_labels:
        print(f"\nMerging data from {processed_subjects} subjects...")
        
        try:
            all_acc_concat = np.concatenate(all_acc, axis=0)
            all_bvp_concat = np.concatenate(all_bvp)
            all_eda_concat = np.concatenate(all_eda)
            all_temp_concat = np.concatenate(all_temp)
            all_labels_concat = np.concatenate(all_labels)
        except ValueError as e:
            print(f"Error during concatenation: {e}")
            print("Shapes of arrays being concatenated:")
            for i, sid in enumerate(subject_dirs): # Assuming order matches all_... lists
                if i < len(all_acc): print(f"  {os.path.basename(sid)} ACC shape: {all_acc[i].shape}")
                # Add similar prints for other modalities if needed
            return # Stop if concatenation fails

        print(f"Cleaned wrist data shape (all at {target_sampling_rate}Hz):")
        print(f"  ACC: {all_acc_concat.shape}")
        print(f"  BVP: {all_bvp_concat.shape}")
        print(f"  EDA: {all_eda_concat.shape}")
        print(f"  TEMP: {all_temp_concat.shape}")
        print(f"  Labels: {all_labels_concat.shape}")
        
        baseline_count = np.sum(all_labels_concat == 1)
        stress_count = np.sum(all_labels_concat == 2)
        print(f"Merged data label distribution: Baseline={baseline_count}, Stress={stress_count}")
        
        merged_dir = os.path.join(output_dir, 'merged')
        os.makedirs(merged_dir, exist_ok=True)
        
        # Save merged data with suffix indicating frequency
        np.save(os.path.join(merged_dir, 'wrist_ACC_700Hz.npy'), all_acc_concat)
        np.save(os.path.join(merged_dir, 'wrist_BVP_700Hz.npy'), all_bvp_concat)
        np.save(os.path.join(merged_dir, 'wrist_EDA_700Hz.npy'), all_eda_concat)
        np.save(os.path.join(merged_dir, 'wrist_TEMP_700Hz.npy'), all_temp_concat)
        np.save(os.path.join(merged_dir, 'labels_700Hz.npy'), all_labels_concat)
        
        with open(os.path.join(merged_dir, 'info.txt'), 'w') as f:
            f.write(f"Number of subjects processed: {processed_subjects}\n")
            f.write(f"Saved Data Sampling Rate (Hz): {target_sampling_rate}\n")
            f.write(f"Baseline samples: {baseline_count}\n")
            f.write(f"Stress samples: {stress_count}\n")
            f.write(f"Total samples: {len(all_labels_concat)}\n")
            f.write(f"ACC shape: {all_acc_concat.shape}\n")
            f.write(f"BVP shape: {all_bvp_concat.shape}\n")
            f.write(f"EDA shape: {all_eda_concat.shape}\n")
            f.write(f"TEMP shape: {all_temp_concat.shape}\n")
            f.write(f"Labels shape: {all_labels_concat.shape}\n")
    else:
        print("No valid wrist data found with labels 1 or 2 across all subjects.")

if __name__ == "__main__":
    # Ensure scipy is installed: pip install scipy
    
    # Configure parameters - ADJUST THESE PATHS AS NEEDED
    data_dir = "/Volumes/xcy/TeamProject/WESAD"  # WESAD original data directory
    # Changed output directory name to reflect 700Hz data
    output_dir = "/Volumes/xcy/TeamProject/WESAD/cleaned_wrist_data_700Hz" 
    
    print("="*80)
    print("WESAD Wrist Data Extraction (with Upsampling to 700Hz)")
    print("="*80)
    
    extract_wesad_wrist_data(data_dir, output_dir)
    
    print("\nWrist data extraction completed!")
    print(f"Cleaned wrist data (at 700Hz) saved in {output_dir}")
    print("="*80) 