#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD Wrist Data Extraction Script
Extracts ACC, BVP, EDA, and TEMP signals from the wrist device,
upsamples them to 700Hz to match label frequency,
and optionally saves either separate modality files (for classical models)
or a combined multi-channel array (for CNNs), plus merged outputs.
"""

import os
import pickle
import numpy as np
from scipy.signal import resample

def extract_wesad_wrist_data(
    data_dir,
    output_dir,
    save_separate=True,
    save_multi=True,
    target_sampling_rate=700
):
    os.makedirs(output_dir, exist_ok=True)

    # find subject folders
    subject_dirs = [
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
           and d.startswith('S') and not d.startswith('S_')
    ]
    print(f"Found {len(subject_dirs)} subject directories")

    # buffers for global merge
    all_acc, all_bvp, all_eda, all_temp = [], [], [], []
    all_labels, all_multi = [], []

    processed_subjects = 0

    for subject_dir in subject_dirs:
        sid = os.path.basename(subject_dir)
        pkl = os.path.join(subject_dir, f"{sid}.pkl")
        if not os.path.exists(pkl):
            print(f"{sid}: no pickle file, skipping")
            continue

        print(f"Processing subject {sid}...")
        try:
            with open(pkl, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print(f"{sid}: load error {e}, skipping")
            continue

        if 'wrist' not in data['signal']:
            print(f"{sid}: no wrist data, skipping")
            continue

        wrist = data['signal']['wrist']
        labels = data['label'].flatten()
        N = len(labels)

        # resample each modality to N samples
        try:
            acc = resample(wrist['ACC'],    N, axis=0)       # (N,3)
            bvp = resample(wrist['BVP'].flatten(), N)       # (N,)
            eda = resample(wrist['EDA'].flatten(), N)       # (N,)
            temp= resample(wrist['TEMP'].flatten(), N)      # (N,)
        except Exception as e:
            print(f"{sid}: resample error {e}, skipping")
            continue

        # mask only labels 1 or 2
        mask = (labels == 1) | (labels == 2)
        if not mask.any():
            print(f"{sid}: no baseline/stress samples, skipping")
            continue

        acc_c    = acc[mask]
        bvp_c    = bvp[mask]
        eda_c    = eda[mask]
        temp_c   = temp[mask]
        labels_c = labels[mask]

        print(f"  Kept {len(labels_c)} samples: "
              f"{(labels_c==1).sum()} baseline, {(labels_c==2).sum()} stress")

        # build multi-channel array (samples, 6)
        wrist_multi = np.column_stack([
            acc_c,                  # 3 ACC channels
            bvp_c[:, None],         # 1 BVP
            eda_c[:, None],         # 1 EDA
            temp_c[:, None]         # 1 TEMP
        ])

        # subject output folder
        out_sub = os.path.join(output_dir, sid)
        os.makedirs(out_sub, exist_ok=True)

        # save separate modalities if requested
        if save_separate:
            np.save(os.path.join(out_sub, 'wrist_ACC_700Hz.npy'), acc_c)
            np.save(os.path.join(out_sub, 'wrist_BVP_700Hz.npy'), bvp_c)
            np.save(os.path.join(out_sub, 'wrist_EDA_700Hz.npy'), eda_c)
            np.save(os.path.join(out_sub, 'wrist_TEMP_700Hz.npy'), temp_c)
            all_acc.append(acc_c)
            all_bvp.append(bvp_c)
            all_eda.append(eda_c)
            all_temp.append(temp_c)

        # save combined multi-channel if requested
        if save_multi:
            np.save(os.path.join(out_sub, 'wrist_multi_700Hz.npy'), wrist_multi)
            all_multi.append(wrist_multi)

        # always save labels
        np.save(os.path.join(out_sub, 'labels_700Hz.npy'), labels_c)
        all_labels.append(labels_c)

        # write subject info
        with open(os.path.join(out_sub, 'info.txt'), 'w') as f:
            f.write(f"Sampling Rate (Hz): {target_sampling_rate}\n")
            f.write(f"Baseline: {(labels_c==1).sum()}, Stress: {(labels_c==2).sum()}\n")
            if save_separate:
                f.write(f"Separate shapes - "
                        f"ACC{acc_c.shape}, BVP{bvp_c.shape}, "
                        f"EDA{eda_c.shape}, TEMP{temp_c.shape}\n")
            if save_multi:
                f.write(f"Multi-channel shape: {wrist_multi.shape}\n")

        processed_subjects += 1

    print(f"\nProcessed {processed_subjects} subjects total")

    # merge and save global outputs
    merged_dir = os.path.join(output_dir, 'merged')
    os.makedirs(merged_dir, exist_ok=True)

    if not all_labels:
        print("No valid wrist data across any subject.")
        return

    labels_all = np.concatenate(all_labels, axis=0)
    np.save(os.path.join(merged_dir, 'labels_700Hz.npy'), labels_all)

    if save_separate:
        np.save(os.path.join(merged_dir, 'wrist_ACC_700Hz.npy'), np.concatenate(all_acc, axis=0))
        np.save(os.path.join(merged_dir, 'wrist_BVP_700Hz.npy'), np.concatenate(all_bvp))
        np.save(os.path.join(merged_dir, 'wrist_EDA_700Hz.npy'), np.concatenate(all_eda))
        np.save(os.path.join(merged_dir, 'wrist_TEMP_700Hz.npy'), np.concatenate(all_temp))
        print(f"Merged separate modalities saved (total samples: {labels_all.shape[0]})")

    if save_multi:
        multi_all = np.concatenate(all_multi, axis=0)
        np.save(os.path.join(merged_dir, 'wrist_multi_700Hz.npy'), multi_all)
        print(f"Merged multi-channel saved (shape: {multi_all.shape})")

    # write merged info
    with open(os.path.join(merged_dir, 'info.txt'), 'w') as f:
        f.write(f"Subjects processed: {processed_subjects}\n")
        f.write(f"Total samples: {labels_all.shape[0]}\n")
        f.write(f"Baseline: {(labels_all==1).sum()}, Stress: {(labels_all==2).sum()}\n")
        if save_separate:
            f.write(f"ACC shape: {np.concatenate(all_acc, axis=0).shape}\n")
            f.write(f"BVP shape: {np.concatenate(all_bvp).shape}\n")
            f.write(f"EDA shape: {np.concatenate(all_eda).shape}\n")
            f.write(f"TEMP shape: {np.concatenate(all_temp).shape}\n")
        if save_multi:
            f.write(f"Multi shape: {multi_all.shape}\n")

if __name__ == "__main__":
    # —— Hard-coded parameters: change these as needed! —— #
    data_dir               = "/kaggle/input/wesaddataset/WESAD"
    output_dir             = "/kaggle/working/WESAD/cleaned_wrist_data_700Hz"
    save_separate          = True   # keep ACC/BVP/EDA/TEMP files
    save_multi             = True    # keep combined multi-channel file
    target_sampling_rate   = 700     # Hz
    # ———————————————————————————————————————————————— #

    extract_wesad_wrist_data(
        data_dir,
        output_dir,
        save_separate=save_separate,
        save_multi=save_multi,
        target_sampling_rate=target_sampling_rate
    )

    print("\nExtraction complete")
    print(f"Data saved in: {output_dir}")
