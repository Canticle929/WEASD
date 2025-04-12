#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD数据清理简化版脚本
仅提取ECG、EDA和Resp信号，只保留标签为1和2的数据
WESAD Data Cleanup Lite script
Only ECG, EDA, and Resp signals are extracted, and only data labels 1 and 2 are retained
"""

import os
import pickle
import numpy as np

def clean_wesad_data(data_dir, output_dir):
    """
    清理WESAD数据，只保留标签为1和2的ECG、EDA和Resp信号
    Clean WESAD data, only retain ECG, EDA, and Resp signals with labels 1 and 2
    
    参数:
        data_dir: WESAD数据目录路径 WESAD data directory path
        output_dir: 清理后数据的输出目录路径 Cleaned data output directory path
    
    """
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # find all subject directories
    subject_dirs = []
    for entry in os.listdir(data_dir):
        entry_path = os.path.join(data_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith('S') and not entry.startswith('S_'):
            subject_dirs.append(entry_path)
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    # create data collection list
    all_ecg = []
    all_eda = []
    all_resp = []
    all_labels = []
    
    # process each subject
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        pickle_file = os.path.join(subject_dir, f"{subject_id}.pkl")
        
        if not os.path.exists(pickle_file):
            print(f"Subject {subject_id} data file {pickle_file} does not exist, skipping")
            continue
            
        print(f"Processing subject {subject_id}...")
        
        # load data
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"Failed to load subject {subject_id} data: {e}")
            continue
        
        # extract chest device ECG, EDA, Resp signals and labels
        chest_data = data['signal']['chest']
        ecg = chest_data['ECG'].flatten()
        eda = chest_data['EDA'].flatten()
        resp = chest_data['Resp'].flatten()
        labels = data['label'].flatten()
        
        # according to WESAD dataset documentation, chest RespiBAN device sampling rate is 700 Hz
        sampling_rate = 700
        
        # only retain samples with labels 1 or 2
        mask = (labels == 1) | (labels == 2)
        
        # check if there are any samples with符合条件的样本
        if not np.any(mask):
            print(f"   Subject {subject_id} has no samples with labels 1 or 2, skipping")
            continue
        
        ecg_clean = ecg[mask]
        eda_clean = eda[mask]
        resp_clean = resp[mask]
        labels_clean = labels[mask]
        
        # statistics of different labels
        baseline_count = np.sum(labels_clean == 1)
        stress_count = np.sum(labels_clean == 2)
        print(f"   Baseline samples: {baseline_count}, Stress samples: {stress_count}")
        
        # save the cleaned data for this subject
        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        np.save(os.path.join(subject_output_dir, 'chest_ECG.npy'), ecg_clean)
        np.save(os.path.join(subject_output_dir, 'chest_EDA.npy'), eda_clean)
        np.save(os.path.join(subject_output_dir, 'chest_Resp.npy'), resp_clean)
        np.save(os.path.join(subject_output_dir, 'labels.npy'), labels_clean)
        
        # record sampling rate
        with open(os.path.join(subject_output_dir, 'info.txt'), 'w') as f:
            f.write(f"Sampling rate: {sampling_rate} Hz\n")
            f.write(f"Baseline samples: {baseline_count}\n")
            f.write(f"Stress samples: {stress_count}\n")
        
        # add the cleaned data to the global list
        all_ecg.append(ecg_clean)
        all_eda.append(eda_clean)
        all_resp.append(resp_clean)
        all_labels.append(labels_clean)
    
    # if there is valid data, merge all subjects' data and save
    if all_ecg:
        print("\nMerging all subjects' data...")
        
        # merge all subjects' data
        all_ecg_concat = np.concatenate(all_ecg)
        all_eda_concat = np.concatenate(all_eda)
        all_resp_concat = np.concatenate(all_resp)
        all_labels_concat = np.concatenate(all_labels)
        
        # print the shape of merged data
        print("Cleaned data shape:")
        print(f"  ECG: {all_ecg_concat.shape}")
        print(f"  EDA: {all_eda_concat.shape}")
        print(f"  Resp: {all_resp_concat.shape}")
        print(f"  标签: {all_labels_concat.shape}")
        
        # print the label distribution
        baseline_count = np.sum(all_labels_concat == 1)
        stress_count = np.sum(all_labels_concat == 2)
        print(f"Merged data label distribution: Baseline={baseline_count}, Stress={stress_count}")
        
        # save the merged data
        merged_dir = os.path.join(output_dir, 'merged')
        os.makedirs(merged_dir, exist_ok=True)
        
        np.save(os.path.join(merged_dir, 'chest_ECG.npy'), all_ecg_concat)
        np.save(os.path.join(merged_dir, 'chest_EDA.npy'), all_eda_concat)
        np.save(os.path.join(merged_dir, 'chest_Resp.npy'), all_resp_concat)
        np.save(os.path.join(merged_dir, 'labels.npy'), all_labels_concat)
        
        # record data statistics
        with open(os.path.join(merged_dir, 'info.txt'), 'w') as f:
            f.write(f"Number of subjects: {len(subject_dirs)}\n")
            f.write(f"Baseline samples: {baseline_count}\n")
            f.write(f"Stress samples: {stress_count}\n")
            f.write(f"Total samples: {len(all_labels_concat)}\n")
            f.write(f"ECG shape: {all_ecg_concat.shape}\n")
            f.write(f"EDA shape: {all_eda_concat.shape}\n")
            f.write(f"Resp shape: {all_resp_concat.shape}\n")
            f.write(f"Labels shape: {all_labels_concat.shape}\n")
    else:
        print("No data found with labels 1 or 2")

if __name__ == "__main__":
    # configure parameters
    data_dir = "/Volumes/xcy/TeamProject/WESAD"  # WESAD original data directory
    output_dir = "/Volumes/xcy/TeamProject/WESAD/cleaned_data"  # cleaned data output directory
    
    print("="*80)
    print("WESAD data cleanup (simplified version)")
    print("="*80)
    
    # 清理数据
    clean_wesad_data(data_dir, output_dir)
    
    print("\nData cleanup completed!")
    print(f"Cleaned data saved in {output_dir}")
    print("="*80) 