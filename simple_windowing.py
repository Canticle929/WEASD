#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD data windowing script
The cleaned sequential signal is split into fixed-size windows for subsequent feature extraction and machine learning

"""

import os
import numpy as np
from scipy import stats

def create_windows(signals, labels, window_size, step_size, sampling_rate):
    """
    Split signals into fixed-size windows
    
    Parameters:
        signals: signal dictionary, format: {signal_name: signal_data}
        labels: label vector
        window_size: window size (seconds)
        step_size: window step size (seconds)
        sampling_rate: sampling rate (Hz)
    
    Returns:
        windows_dict: windowed signal dictionary, format: {signal_name: [windows]}
        window_labels: label for each window
    """
    # calculate the number of samples per window
    samples_per_window = int(window_size * sampling_rate)
    step_samples = int(step_size * sampling_rate)
    
    # initialize window dictionary and label list
    windows_dict = {signal_name: [] for signal_name in signals}
    window_labels = []
    
    # calculate the number of windows that can be created
    signal_length = min(len(signal_data) for signal_data in signals.values())
    n_windows = (signal_length - samples_per_window) // step_samples + 1
    
    if n_windows <= 0:
        print(f"Warning: signal length ({signal_length}) is less than window size ({samples_per_window}), cannot create windows")
        return windows_dict, np.array([])
    
    print(f"signal length: {signal_length} samples")
    print(f"window size: {samples_per_window} samples ({window_size} seconds)")
    print(f"window step: {step_samples} samples ({step_size} seconds)")
    print(f"can create {n_windows} windows")
    
    # create windows
    for i in range(n_windows):
        start = i * step_samples
        end = start + samples_per_window
        
        # get the label of the current window (use mode as window label)
        window_labels_segment = labels[start:end]
        window_label = stats.mode(window_labels_segment, keepdims=True)[0][0]
        
        # create a window for each signal
        for signal_name, signal_data in signals.items():
            windows_dict[signal_name].append(signal_data[start:end])
        
        window_labels.append(window_label)
    
    # convert window list to numpy array
    for signal_name in windows_dict:
        windows_dict[signal_name] = np.array(windows_dict[signal_name])
    
    return windows_dict, np.array(window_labels)

def window_subject_data(subject_dir, output_dir, window_size=5, step_size=2.5):
    """
    Process the data of a single subject
    
    Parameters:
        subject_dir: directory of cleaned data for a subject
        output_dir: output directory for windowed data
        window_size: window size (seconds)
        step_size: window step size (seconds)
    
    Returns:
        success: whether the data is processed successfully
    """
    # extract the subject ID
    subject_id = os.path.basename(subject_dir)
    
    # load signals and labels
    try:
        ecg = np.load(os.path.join(subject_dir, "chest_ECG.npy"))
        eda = np.load(os.path.join(subject_dir, "chest_EDA.npy"))
        resp = np.load(os.path.join(subject_dir, "chest_Resp.npy"))
        labels = np.load(os.path.join(subject_dir, "labels.npy"))
    except Exception as e:
        print(f"Error loading data for subject {subject_id}: {e}")
        return False
    
    # read sampling rate
    try:
        with open(os.path.join(subject_dir, "info.txt"), "r") as f:
            for line in f:
                if line.startswith("Sampling rate:"):
                    sampling_rate = float(line.split(":")[1].strip().split(" ")[0])
                    break
    except:
        print(f"无法读取采样率，使用默认值 700 Hz")
        sampling_rate = 700  # default sampling rate for chest device of WESAD
    
    # check the consistency of data length
    if len(ecg) != len(eda) or len(ecg) != len(resp) or len(ecg) != len(labels):
        print(f"Warning: data length for subject {subject_id} is inconsistent")
        min_length = min(len(ecg), len(eda), len(resp), len(labels))
        ecg = ecg[:min_length]
        eda = eda[:min_length]
        resp = resp[:min_length]
        labels = labels[:min_length]
    
    # check if there is enough data to create windows
    if len(ecg) < window_size * sampling_rate:
        print(f"Warning: data length for subject {subject_id} ({len(ecg)}) is less than window size ({window_size * sampling_rate}), skipping")
        return False
    
    print(f"Processing subject {subject_id}...")
    
    # create signal dictionary
    signals = {
        "chest_ECG": ecg,
        "chest_EDA": eda,
        "chest_Resp": resp
    }
    
    # create windows
    windows_dict, window_labels = create_windows(signals, labels, window_size, step_size, sampling_rate)
    
    # check if any windows are created
    if len(window_labels) == 0:
        print(f"Warning: no windows created for subject {subject_id}, skipping")
        return False
    
    # statistics of label distribution
    unique_labels, counts = np.unique(window_labels, return_counts=True)
    label_distribution = {label: count for label, count in zip(unique_labels, counts)}
    print(f"Window label distribution: {label_distribution}")
    
    # create output directory
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # save windowed data
    for signal_name, windows in windows_dict.items():
        np.save(os.path.join(subject_output_dir, f"{signal_name}_windows.npy"), windows)
    
    np.save(os.path.join(subject_output_dir, "window_labels.npy"), window_labels)
    
    # save window information
    with open(os.path.join(subject_output_dir, "window_info.txt"), "w") as f:
        f.write(f"Subject: {subject_id}\n")
        f.write(f"Window size: {window_size} seconds\n")
        f.write(f"Window step: {step_size} seconds\n")
        f.write(f"Sampling rate: {sampling_rate} Hz\n")
        f.write(f"Number of samples per window: {int(window_size * sampling_rate)}\n")
        f.write(f"Total number of windows: {len(window_labels)}\n\n")
        
        f.write("Label distribution:\n")
        for label, count in zip(unique_labels, counts):
            label_name = "Baseline" if label == 1 else "Stress" if label == 2 else f"Unknown ({label})"
            f.write(f"   Label {label} ({label_name}): {count} windows ({count/len(window_labels)*100:.2f}%)\n")
        
        f.write("\nWindow shape:\n")
        for signal_name, windows in windows_dict.items():
            f.write(f"  {signal_name}: {windows.shape}\n")
    
    print(f"Subject {subject_id} processed successfully, created {len(window_labels)} windows")
    return True

def window_all_subjects(input_dir, output_dir, window_size=5, step_size=2.5):
    """
    Process all subjects' data for windowing
    
    Parameters:
        input_dir: directory of cleaned data
        output_dir: output directory for windowed data
        window_size: window size (seconds)
        step_size: window step size (seconds)
    """
    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # find all subject directories
    subject_dirs = []
    for entry in os.listdir(input_dir):
        entry_path = os.path.join(input_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("S") and entry != "merged":
            subject_dirs.append(entry_path)
    
    print(f"Found {len(subject_dirs)} subject directories")
    
    # process data for each subject
    successful = 0
    for subject_dir in subject_dirs:
        if window_subject_data(subject_dir, output_dir, window_size, step_size):
            successful += 1
    
    print(f"Successfully processed {successful}/{len(subject_dirs)} subjects' data")
    
    # merge all subjects' windowed data
    merge_windowed_data(output_dir)

def merge_windowed_data(windowed_data_dir):
    """
    Merge all subjects' windowed data
    
    Parameters:
        windowed_data_dir: directory of windowed data
    """
    print("\nMerging all subjects' windowed data...")
    
    # initialize dictionary for merged data
    all_windows = {}
    all_labels = []
    all_subjects = []
    
    # find all subject directories
    subject_dirs = []
    for entry in os.listdir(windowed_data_dir):
        entry_path = os.path.join(windowed_data_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("S"):
            subject_dirs.append(entry_path)
    
    # process data for each subject
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        
        # load window labels
        labels_path = os.path.join(subject_dir, "window_labels.npy")
        if not os.path.exists(labels_path):
            print(f"Skipping subject {subject_id}: window labels file not found")
            continue
        
        window_labels = np.load(labels_path)
        
        # find all window data files
        window_files = [f for f in os.listdir(subject_dir) if f.endswith("_windows.npy")]
        
        # load window data for each signal
        for window_file in window_files:
            signal_name = window_file.replace("_windows.npy", "")
            windows = np.load(os.path.join(subject_dir, window_file))
            
            # add windows to merged data
            if signal_name not in all_windows:
                all_windows[signal_name] = []
            
            all_windows[signal_name].append(windows)
        
        # add labels and subject ID
        all_labels.append(window_labels)
        all_subjects.extend([subject_id] * len(window_labels))
    
    # if no data is found, return
    if not all_windows:
        print("No window data found")
        return
    
    # merge data
    merged_windows = {}
    for signal_name, windows_list in all_windows.items():
        merged_windows[signal_name] = np.vstack(windows_list)
    
    merged_labels = np.concatenate(all_labels)
    merged_subjects = np.array(all_subjects)
    
    # create output directory for merged data
    merged_dir = os.path.join(windowed_data_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    # save merged data
    for signal_name, windows in merged_windows.items():
        np.save(os.path.join(merged_dir, f"{signal_name}_windows.npy"), windows)
    
    np.save(os.path.join(merged_dir, "window_labels.npy"), merged_labels)
    np.save(os.path.join(merged_dir, "subject_ids.npy"), merged_subjects)
    
    # save merged data information
    with open(os.path.join(merged_dir, "merged_info.txt"), "w") as f:
        f.write("Merged window data information\n")
        f.write("============================\n\n")
        f.write(f"Number of subjects: {len(subject_dirs)}\n")
        f.write(f"Total number of windows: {len(merged_labels)}\n\n")
        
        # label distribution
        unique_labels, counts = np.unique(merged_labels, return_counts=True)
        f.write("Label distribution:\n")
        for label, count in zip(unique_labels, counts):
            label_name = "Baseline" if label == 1 else "Stress" if label == 2 else f"Unknown ({label})"
            f.write(f"   Label {label} ({label_name}): {count} windows ({count/len(merged_labels)*100:.2f}%)\n")
        
        # number of windows for each subject
        f.write("\nNumber of windows for each subject:\n")
        unique_subjects, counts = np.unique(merged_subjects, return_counts=True)
        for subject, count in zip(unique_subjects, counts):
            f.write(f"  {subject}: {count} windows\n")
        
        # window shape
        f.write("\nWindow shape:\n")
        for signal_name, windows in merged_windows.items():
            f.write(f"  {signal_name}: {windows.shape}\n")
    
    print(f"Merged successfully, total {len(merged_labels)} windows, saved to {merged_dir}")
    print(f"Label distribution: {dict(zip(unique_labels, counts))}")

if __name__ == "__main__":
    # configure parameters
    input_dir = "/Volumes/xcy/TeamProject/WESAD/cleaned_data"  # cleaned data directory
    output_dir = "/Volumes/xcy/TeamProject/WESAD/windowed_data"  # windowed data output directory
    window_size = 5  # window size (seconds)
    step_size = 2.5  # window step size (seconds)
    
    print("="*80)
    print("WESAD data windowing processing")
    print("="*80)
    
    # 窗口化处理
    window_all_subjects(input_dir, output_dir, window_size, step_size)
    
    print("\nWindowing processing completed!")
    print(f"Windowed data saved in {output_dir}")
    print("="*80) 