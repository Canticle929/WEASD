#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD data normalization and alignment script
将不同采样率的信号降采样到统一频率(4Hz)，并进行Z-score标准化
"""

import os
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt

def normalize_and_align(input_dir, output_dir, fs_target=4):
    """
    Downsample cleaned data to a uniform frequency and normalize it
    
    Parameters:
        input_dir: path to the cleaned data directory
        output_dir: path to the normalized data output directory
        fs_target: target sampling rate, default is 4Hz
    """
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # find all subject directories
    subject_dirs = []
    for entry in os.listdir(input_dir):
        entry_path = os.path.join(input_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith('S') and entry != 'merged':
            subject_dirs.append(entry_path)
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    # create data collection list
    all_ecg_ds = []
    all_eda_ds = []
    all_resp_ds = []
    all_labels_ds = []
    
    # process data for each subject
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        print(f"Processing subject {subject_id}...")
        
        try:
            # load cleaned data
            ecg = np.load(os.path.join(subject_dir, 'chest_ECG.npy'))
            eda = np.load(os.path.join(subject_dir, 'chest_EDA.npy'))
            resp = np.load(os.path.join(subject_dir, 'chest_Resp.npy'))
            labels = np.load(os.path.join(subject_dir, 'labels.npy'))
            
            # read sampling rate
            fs_orig = 700  # sampling rate of chest device
            
            # calculate decimation factor
            decimation_factor = fs_orig // fs_target
            
            # ensure length is enough for decimation
            min_length = decimation_factor * (len(ecg) // decimation_factor)
            if min_length < decimation_factor:
                print(f" subject {subject_id} data length is not enough for decimation, skipping")
                continue
                
            # truncate data to ensure it can be divided by the decimation factor
            ecg = ecg[:min_length]
            eda = eda[:min_length]
            resp = resp[:min_length]
            labels = labels[:min_length]
            
            # downsample signals
            ecg_ds = decimate(ecg, decimation_factor)
            eda_ds = decimate(eda, decimation_factor)
            resp_ds = decimate(resp, decimation_factor)
            
            # downsample labels (simply take the label at the corresponding position after decimation)
            labels_ds = labels[::decimation_factor]
            
            # normalize each signal by Z-score
            ecg_mean, ecg_std = ecg_ds.mean(), ecg_ds.std()
            eda_mean, eda_std = eda_ds.mean(), eda_ds.std()
            resp_mean, resp_std = resp_ds.mean(), resp_ds.std()
            
            ecg_norm = (ecg_ds - ecg_mean) / ecg_std
            eda_norm = (eda_ds - eda_mean) / eda_std
            resp_norm = (resp_ds - resp_mean) / resp_std
            
            print(f"  original signal length: {len(ecg)}, decimated signal length: {len(ecg_ds)}")
            print(f"  ECG normalized range: {ecg_norm.min():.2f} ~ {ecg_norm.max():.2f}")
            print(f"  EDA normalized range: {eda_norm.min():.2f} ~ {eda_norm.max():.2f}")
            print(f"  Resp归一化范围: {resp_norm.min():.2f} ~ {resp_norm.max():.2f}")
            
            # create subject output directory
            subject_output_dir = os.path.join(output_dir, subject_id)
            os.makedirs(subject_output_dir, exist_ok=True)
            
            # save normalized data
            np.save(os.path.join(subject_output_dir, 'chest_ECG_norm.npy'), ecg_norm)
            np.save(os.path.join(subject_output_dir, 'chest_EDA_norm.npy'), eda_norm)
            np.save(os.path.join(subject_output_dir, 'chest_Resp_norm.npy'), resp_norm)
            np.save(os.path.join(subject_output_dir, 'labels_ds.npy'), labels_ds)
            
            # save normalization parameters,便于之后应用于新数据
            norm_params = {
                'ecg_mean': ecg_mean, 'ecg_std': ecg_std,
                'eda_mean': eda_mean, 'eda_std': eda_std,
                'resp_mean': resp_mean, 'resp_std': resp_std,
                'fs_orig': fs_orig, 'fs_target': fs_target,
                'decimation_factor': decimation_factor
            }
            np.save(os.path.join(subject_output_dir, 'norm_params.npy'), norm_params)
            
            # record information
            with open(os.path.join(subject_output_dir, 'norm_info.txt'), 'w') as f:
                f.write(f"subject: {subject_id}\n")
                f.write(f"original sampling rate: {fs_orig} Hz\n")
                f.write(f"target sampling rate: {fs_target} Hz\n")
                f.write(f"decimation factor: {decimation_factor}\n")
                f.write(f"original signal length: {len(ecg)} samples\n")
                f.write(f"decimated signal length: {len(ecg_ds)} samples\n\n")
                
                f.write("normalization parameters:\n")
                f.write(f"  ECG mean: {ecg_mean:.4f}, std: {ecg_std:.4f}\n")
                f.write(f"  EDA mean: {eda_mean:.4f}, std: {eda_std:.4f}\n")
                f.write(f"  Resp mean: {resp_mean:.4f}, std: {resp_std:.4f}\n\n")
                
                f.write("signal range:\n")
                f.write(f"  ECG: {ecg_norm.min():.4f} ~ {ecg_norm.max():.4f}\n")
                f.write(f"  EDA: {eda_norm.min():.4f} ~ {eda_norm.max():.4f}\n")
                f.write(f"  Resp: {resp_norm.min():.4f} ~ {resp_norm.max():.4f}\n\n")
                
                baseline_count = np.sum(labels_ds == 1)
                stress_count = np.sum(labels_ds == 2)
                f.write(f"label distribution:\n")
                f.write(f"  baseline: {baseline_count} ({baseline_count/len(labels_ds)*100:.2f}%)\n")
                f.write(f"  stress: {stress_count} ({stress_count/len(labels_ds)*100:.2f}%)\n")
            
            # plot normalized signal comparison
            plt.figure(figsize=(15, 10))
            
            # plot only the first 1000 original samples and the corresponding decimated samples
            orig_samples = 1000
            ds_samples = orig_samples // decimation_factor
            
            # ECG original and normalized comparison
            plt.subplot(3, 2, 1)
            plt.plot(ecg[:orig_samples])
            plt.title(f'ECG original signal ({fs_orig} Hz)')
            plt.grid(True)
            
            plt.subplot(3, 2, 2)
            plt.plot(ecg_norm[:ds_samples])
            plt.title(f'ECG normalized signal ({fs_target} Hz)')
            plt.grid(True)
            
            # EDA original and normalized comparison
            plt.subplot(3, 2, 3)
            plt.plot(eda[:orig_samples])
            plt.title(f'EDA original signal ({fs_orig} Hz)')
            plt.grid(True)
            
            plt.subplot(3, 2, 4)
            plt.plot(eda_norm[:ds_samples])
            plt.title(f'EDA normalized signal ({fs_target} Hz)')
            plt.grid(True)
            
            # Resp original and normalized comparison
            plt.subplot(3, 2, 5)
            plt.plot(resp[:orig_samples])
            plt.title(f'Resp original signal ({fs_orig} Hz)')
            plt.grid(True)
            
            plt.subplot(3, 2, 6)
            plt.plot(resp_norm[:ds_samples])
            plt.title(f'Resp normalized signal ({fs_target} Hz)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(subject_output_dir, 'signal_comparison.png'), dpi=300)
            plt.close()
            
            # add normalized data to the global list
            all_ecg_ds.append(ecg_norm)
            all_eda_ds.append(eda_norm)
            all_resp_ds.append(resp_norm)
            all_labels_ds.append(labels_ds)
            
        except Exception as e:
            print(f"  process subject {subject_id} error: {e}")
            continue
    
    # if there is valid data, merge and save
    if all_ecg_ds:
        print("\nmerge all subjects' normalized data...")
        
        # merge all subjects' data
        all_ecg_ds_concat = np.concatenate(all_ecg_ds)
        all_eda_ds_concat = np.concatenate(all_eda_ds)
        all_resp_ds_concat = np.concatenate(all_resp_ds)
        all_labels_ds_concat = np.concatenate(all_labels_ds)
        
        # print merged data shape
        print("normalized data shape:")
        print(f"  ECG: {all_ecg_ds_concat.shape}")
        print(f"  EDA: {all_eda_ds_concat.shape}")
        print(f"  Resp: {all_resp_ds_concat.shape}")
        print(f"  labels: {all_labels_ds_concat.shape}")
        
        # create merged data output directory
        merged_dir = os.path.join(output_dir, 'merged')
        os.makedirs(merged_dir, exist_ok=True)
        
        # save merged data
        np.save(os.path.join(merged_dir, 'chest_ECG_norm.npy'), all_ecg_ds_concat)
        np.save(os.path.join(merged_dir, 'chest_EDA_norm.npy'), all_eda_ds_concat)
        np.save(os.path.join(merged_dir, 'chest_Resp_norm.npy'), all_resp_ds_concat)
        np.save(os.path.join(merged_dir, 'labels_ds.npy'), all_labels_ds_concat)
        
        # calculate merged label distribution
        baseline_count = np.sum(all_labels_ds_concat == 1)
        stress_count = np.sum(all_labels_ds_concat == 2)
        
        # record merged data information
        with open(os.path.join(merged_dir, 'merged_norm_info.txt'), 'w') as f:
            f.write("merged normalized data information\n")
            f.write("===============================\n\n")
            f.write(f"number of subjects: {len(subject_dirs)}\n")
            f.write(f"original sampling rate: {fs_orig} Hz\n")
            f.write(f"target sampling rate: {fs_target} Hz\n")
            f.write(f"decimation factor: {decimation_factor}\n\n")
            
            f.write("data shape:\n")
            f.write(f"  ECG: {all_ecg_ds_concat.shape}\n")
            f.write(f"  EDA: {all_eda_ds_concat.shape}\n")
            f.write(f"  Resp: {all_resp_ds_concat.shape}\n")
            f.write(f"  labels: {all_labels_ds_concat.shape}\n\n")
            
            f.write("label distribution:\n")
            f.write(f"  baseline: {baseline_count} ({baseline_count/len(all_labels_ds_concat)*100:.2f}%)\n")
            f.write(f"  stress: {stress_count} ({stress_count/len(all_labels_ds_concat)*100:.2f}%)\n\n")
            
            f.write("signal range:\n")
            f.write(f"  ECG: {all_ecg_ds_concat.min():.4f} ~ {all_ecg_ds_concat.max():.4f}\n")
            f.write(f"  EDA: {all_eda_ds_concat.min():.4f} ~ {all_eda_ds_concat.max():.4f}\n")
            f.write(f"  Resp: {all_resp_ds_concat.min():.4f} ~ {all_resp_ds_concat.max():.4f}\n")
    else:
        print("no valid data found for processing")

if __name__ == "__main__":
    # configure parameters
    input_dir = "/Volumes/xcy/TeamProject/WESAD/cleaned_data"  # cleaned data directory
    output_dir = "/Volumes/xcy/TeamProject/WESAD/normalized_data"  # normalized data output directory
    fs_target = 4  # target sampling rate, unit Hz
    
    print("="*80)
    print("WESAD data normalization and alignment")
    print("="*80)
    
    # data normalization and alignment
    normalize_and_align(input_dir, output_dir, fs_target)
    
    print("\ndata normalization and alignment completed!")
    print(f"normalized data saved in {output_dir}")
    print("="*80) 