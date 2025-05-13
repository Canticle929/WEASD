#
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD Wrist Data Windowing Script (Flexible for Classical & CNN)
- Can consume either separate modality files or a combined multi-channel array.
- Outputs windowed arrays accordingly.
"""
import os
import numpy as np
import glob
from scipy import stats
import matplotlib.pyplot as plt


def window_wrist_data(
    input_dir,
    output_dir,
    window_size=5,
    overlap=0.5,
    use_separate=True,
    use_multi=True,
    fs=700
):
    """
    Args:
        input_dir (str): Directory containing subject subfolders with cleaned data.
        output_dir (str): Directory to save windowed outputs.
        window_size (float): Window length in seconds.
        overlap (float): Fractional overlap between windows (0 to <1).
        use_separate (bool): If True, load separate modality files (ACC, BVP, EDA, TEMP).
        use_multi (bool): If True, load combined multi-channel arrays.
        fs (int): Sampling rate (Hz).
    """
    os.makedirs(output_dir, exist_ok=True)

    win_samples = int(window_size * fs)
    step = int(win_samples * (1 - overlap))

    # gather subject dirs
    subj_dirs = [d for d in os.listdir(input_dir)
                 if os.path.isdir(os.path.join(input_dir, d)) and d != 'merged']
    print(f"Found {len(subj_dirs)} subjects to window")

    # global buffers
    all_separate = {ch: [] for ch in ['ACC','BVP','EDA','TEMP']}
    all_multi = []
    all_labels = []

    for sid in subj_dirs:
        subj_in = os.path.join(input_dir, sid)
        subj_out = os.path.join(output_dir, sid)
        os.makedirs(subj_out, exist_ok=True)
        print(f"Windowing subject {sid}...")

        # load data
        if use_multi and os.path.exists(os.path.join(subj_in,'wrist_multi_700Hz.npy')):
            data = np.load(os.path.join(subj_in,'wrist_multi_700Hz.npy'))  # (N, C)
            labels = np.load(os.path.join(subj_in,'labels_700Hz.npy'))     # (N,)
            n_channels = data.shape[1]
        elif use_separate:
            acc = np.load(os.path.join(subj_in,'wrist_ACC_700Hz.npy'))    # (N,3)
            bvp = np.load(os.path.join(subj_in,'wrist_BVP_700Hz.npy'))    # (N,)
            eda = np.load(os.path.join(subj_in,'wrist_EDA_700Hz.npy'))    # (N,)
            temp = np.load(os.path.join(subj_in,'wrist_TEMP_700Hz.npy'))  # (N,)
            labels = np.load(os.path.join(subj_in,'labels_700Hz.npy'))    # (N,)
            # stack locally for multi-windowing
            data = np.column_stack([acc, bvp[:,None], eda[:,None], temp[:,None]])
            n_channels = data.shape[1]
        else:
            raise ValueError("No input data found for subject: %s" % sid)

        N = len(labels)
        num_win = (N - win_samples) // step + 1
        print(f"  {num_win} windows: {window_size}s at {fs}Hz, step {step/fs}s")

        # per-subject buffers
        sep_windows = {ch: [] for ch in ['ACC','BVP','EDA','TEMP']}
        multi_windows = []
        win_labels = []

        for i in range(num_win):
            start = i * step
            end = start + win_samples
            seg = data[start:end]  # shape (win_samples, n_channels)
            seg_lbl = labels[start:end]
            lbl = stats.mode(seg_lbl, keepdims=False)[0]
            
            # append multi-channel window
            multi_windows.append(seg)
            win_labels.append(lbl)

            # append separate modality windows
            sep_windows['ACC'].append(seg[:, :3])
            sep_windows['BVP'].append(seg[:, 3])
            sep_windows['EDA'].append(seg[:, 4])
            sep_windows['TEMP'].append(seg[:, 5])

        # convert to arrays
        multi_arr = np.stack(multi_windows)
        labels_arr = np.array(win_labels)
        sep_arrs = {ch: np.stack(sep_windows[ch]) for ch in sep_windows}

        # save per-subject
        if use_multi:
            np.save(os.path.join(subj_out,'X_windows.npy'), multi_arr)
        if use_separate:
            for ch in sep_arrs:
                np.save(os.path.join(subj_out,f'{ch}_windows.npy'), sep_arrs[ch])
        np.save(os.path.join(subj_out,'y_windows.npy'), labels_arr)

        # append to global
        all_multi.append(multi_arr)
        all_labels.append(labels_arr)
        for ch in sep_arrs:
            all_separate[ch].append(sep_arrs[ch])

        # summary
        print(f"  Saved {multi_arr.shape[0]} windows for {sid}")

    # merge global
    merged = os.path.join(output_dir,'merged')
    os.makedirs(merged, exist_ok=True)
    all_labels_cat = np.concatenate(all_labels)
    np.save(os.path.join(merged,'y_windows_all.npy'), all_labels_cat)

    if use_multi:
        X_all = np.concatenate(all_multi, axis=0)
        np.save(os.path.join(merged,'X_windows_all.npy'), X_all)
        print(f"Merged multi: {X_all.shape}")

    if use_separate:
        for ch in all_separate:
            arr = np.concatenate(all_separate[ch], axis=0)
            np.save(os.path.join(merged,f'{ch}_windows_all.npy'), arr)
            print(f"Merged {ch}: {arr.shape}")

    print("Windowing complete. Merged data in:", merged)


if __name__ == '__main__':
    # Configuring parameters 
    input_dir    = "/kaggle/working/WESAD/cleaned_wrist_data_700Hz"
    output_dir   = "/kaggle/working/WESAD/windowed_wrist_data"
    window_size  = 5      
    overlap      = 0.5    
    use_separate = True   
    use_multi    = True   
    fs           = 700    

    window_wrist_data(
        input_dir,
        output_dir,
        window_size,
        overlap,
        use_separate,
        use_multi,
        fs
    )

    print("\nWindowing complete!")
    print(f"Windowed data saved in: {output_dir}")