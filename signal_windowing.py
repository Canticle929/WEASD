#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WESAD数据窗口化脚本（滑动窗口方法）
将标准化后的信号切分成固定长度的窗口，确保窗口内仅包含同一种状态的信号
使用60秒窗口长度，30秒步长（50%重叠率）
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def create_windows(signals, labels, window_size_sec, step_size_sec, fs):
    """
    将信号切分成固定大小的窗口，确保每个窗口只包含同一种状态的信号
    
    参数:
        signals: 信号字典，格式为 {signal_name: signal_data}
        labels: 标签向量
        window_size_sec: 窗口大小（秒）
        step_size_sec: 窗口步长（秒）
        fs: 采样率（Hz）
    
    返回:
        windows_dict: 窗口化后的信号字典，格式为 {signal_name: [windows]}
        window_labels: 每个窗口的标签
    """
    # 计算每个窗口的样本数和步长
    window_size = int(window_size_sec * fs)
    step_size = int(step_size_sec * fs)
    
    # 初始化窗口字典和标签列表
    windows_dict = {signal_name: [] for signal_name in signals}
    window_labels = []
    
    # 记录标签变化点
    label_change_indices = [0]  # 从信号开始位置算起
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            label_change_indices.append(i)
    label_change_indices.append(len(labels))  # 添加信号结束位置
    
    print(f"标签变化点位置: {len(label_change_indices)}")
    
    # 在每个连续标签段内创建窗口
    for i in range(len(label_change_indices) - 1):
        start_idx = label_change_indices[i]
        end_idx = label_change_indices[i+1]
        segment_length = end_idx - start_idx
        
        # 当前段的标签
        segment_label = labels[start_idx]
        
        # 如果片段长度小于窗口大小，跳过
        if segment_length < window_size:
            print(f"片段长度 ({segment_length} 样本点) 小于窗口大小 ({window_size} 样本点)，跳过")
            continue
        
        # 在当前段内创建窗口
        for j in range(0, segment_length - window_size + 1, step_size):
            seg_start = start_idx + j
            seg_end = seg_start + window_size
            
            # 为每个信号创建窗口
            for signal_name, signal_data in signals.items():
                windows_dict[signal_name].append(signal_data[seg_start:seg_end])
            
            # 添加窗口标签
            window_labels.append(segment_label)
    
    # 将窗口列表转换为numpy数组
    for signal_name in windows_dict:
        if windows_dict[signal_name]:  # 确保列表不为空
            windows_dict[signal_name] = np.array(windows_dict[signal_name])
        else:
            windows_dict[signal_name] = np.array([])
    
    window_labels = np.array(window_labels)
    
    print(f"创建了 {len(window_labels)} 个窗口")
    
    return windows_dict, window_labels

def window_subject_data(subject_dir, output_dir, window_size_sec=60, step_size_sec=30):
    """
    对单个受试者的标准化数据进行窗口化处理
    
    参数:
        subject_dir: 受试者标准化后数据的目录
        output_dir: 窗口化后数据的输出目录
        window_size_sec: 窗口大小（秒）
        step_size_sec: 窗口步长（秒）
    
    返回:
        success: 是否成功处理
    """
    # 提取受试者ID
    subject_id = os.path.basename(subject_dir)
    
    # 加载标准化后的信号和标签
    try:
        ecg = np.load(os.path.join(subject_dir, "chest_ECG_norm.npy"))
        eda = np.load(os.path.join(subject_dir, "chest_EDA_norm.npy"))
        resp = np.load(os.path.join(subject_dir, "chest_Resp_norm.npy"))
        labels = np.load(os.path.join(subject_dir, "labels_ds.npy"))
    except Exception as e:
        print(f"加载受试者 {subject_id} 的数据时出错: {e}")
        return False
    
    # 获取采样率信息
    try:
        norm_params = np.load(os.path.join(subject_dir, "norm_params.npy"), allow_pickle=True).item()
        fs = norm_params['fs_target']
    except:
        print(f"无法读取采样率，使用默认值 4 Hz")
        fs = 4  # 默认降采样后的采样率
    
    print(f"处理受试者 {subject_id}...")
    print(f"信号长度: {len(ecg)} 样本点")
    print(f"采样率: {fs} Hz")
    print(f"窗口大小: {window_size_sec} 秒 ({window_size_sec * fs} 样本点)")
    print(f"窗口步长: {step_size_sec} 秒 ({step_size_sec * fs} 样本点)")
    
    # 检查标签
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    print(f"标签分布: {dict(zip(unique_labels, label_counts))}")
    
    # 创建信号字典
    signals = {
        "chest_ECG_norm": ecg,
        "chest_EDA_norm": eda,
        "chest_Resp_norm": resp
    }
    
    # 创建窗口
    windows_dict, window_labels = create_windows(signals, labels, window_size_sec, step_size_sec, fs)
    
    # 检查是否成功创建窗口
    if len(window_labels) == 0:
        print(f"受试者 {subject_id} 没有创建任何窗口，跳过")
        return False
    
    # 统计窗口标签分布
    unique_labels, counts = np.unique(window_labels, return_counts=True)
    label_distribution = {label: count for label, count in zip(unique_labels, counts)}
    print(f"窗口标签分布: {label_distribution}")
    
    # 创建输出目录
    subject_output_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_output_dir, exist_ok=True)
    
    # 保存窗口化后的数据
    for signal_name, windows in windows_dict.items():
        if len(windows) > 0:  # 确保有窗口数据
            np.save(os.path.join(subject_output_dir, f"{signal_name}_windows.npy"), windows)
    
    np.save(os.path.join(subject_output_dir, "window_labels.npy"), window_labels)
    
    # 保存窗口信息
    with open(os.path.join(subject_output_dir, "window_info.txt"), "w") as f:
        f.write(f"受试者: {subject_id}\n")
        f.write(f"窗口大小: {window_size_sec} 秒\n")
        f.write(f"窗口步长: {step_size_sec} 秒\n")
        f.write(f"采样率: {fs} Hz\n")
        f.write(f"每个窗口样本数: {int(window_size_sec * fs)}\n")
        f.write(f"总窗口数: {len(window_labels)}\n\n")
        
        f.write("标签分布:\n")
        for label, count in zip(unique_labels, counts):
            label_name = "基线" if label == 1 else "压力" if label == 2 else f"未知 ({label})"
            f.write(f"  标签 {label} ({label_name}): {count} 窗口 ({count/len(window_labels)*100:.2f}%)\n")
        
        f.write("\n窗口形状:\n")
        for signal_name, windows in windows_dict.items():
            if len(windows) > 0:
                f.write(f"  {signal_name}: {windows.shape}\n")
    
    # 可视化一些窗口示例
    if len(window_labels) > 0:
        plt.figure(figsize=(15, 10))
        
        # 为每种标签选择一个窗口进行可视化
        for i, label in enumerate(unique_labels):
            # 找到该标签的第一个窗口
            window_idx = np.where(window_labels == label)[0][0]
            
            # 绘制该窗口的三种信号
            plt.subplot(len(unique_labels), 3, i*3+1)
            plt.plot(windows_dict["chest_ECG_norm"][window_idx])
            plt.title(f'标签 {label} 的ECG窗口')
            plt.grid(True)
            
            plt.subplot(len(unique_labels), 3, i*3+2)
            plt.plot(windows_dict["chest_EDA_norm"][window_idx])
            plt.title(f'标签 {label} 的EDA窗口')
            plt.grid(True)
            
            plt.subplot(len(unique_labels), 3, i*3+3)
            plt.plot(windows_dict["chest_Resp_norm"][window_idx])
            plt.title(f'标签 {label} 的Resp窗口')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(subject_output_dir, "window_examples.png"), dpi=300)
        plt.close()
    
    print(f"受试者 {subject_id} 处理完成，创建了 {len(window_labels)} 个窗口")
    return True

def window_all_subjects(input_dir, output_dir, window_size_sec=60, step_size_sec=30):
    """
    对所有受试者的标准化数据进行窗口化处理
    
    参数:
        input_dir: 标准化后数据的目录
        output_dir: 窗口化后数据的输出目录
        window_size_sec: 窗口大小（秒）
        step_size_sec: 窗口步长（秒）
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有受试者目录
    subject_dirs = []
    for entry in os.listdir(input_dir):
        entry_path = os.path.join(input_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("S") and entry != "merged":
            subject_dirs.append(entry_path)
    
    print(f"找到 {len(subject_dirs)} 个受试者目录")
    
    # 处理每个受试者的数据
    successful = 0
    for subject_dir in subject_dirs:
        if window_subject_data(subject_dir, output_dir, window_size_sec, step_size_sec):
            successful += 1
    
    print(f"共成功处理 {successful}/{len(subject_dirs)} 个受试者的数据")
    
    # 合并所有受试者的窗口数据
    merge_windowed_data(output_dir)

def merge_windowed_data(windowed_data_dir):
    """
    合并所有受试者的窗口化数据
    
    参数:
        windowed_data_dir: 窗口化数据的目录
    """
    print("\n合并所有受试者的窗口数据...")
    
    # 初始化合并数据的字典
    all_windows = {}
    all_labels = []
    all_subjects = []
    
    # 查找所有受试者目录
    subject_dirs = []
    for entry in os.listdir(windowed_data_dir):
        entry_path = os.path.join(windowed_data_dir, entry)
        if os.path.isdir(entry_path) and entry.startswith("S"):
            subject_dirs.append(entry_path)
    
    # 对每个受试者
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        
        # 加载窗口标签
        labels_path = os.path.join(subject_dir, "window_labels.npy")
        if not os.path.exists(labels_path):
            print(f"跳过受试者 {subject_id}：没有找到窗口标签文件")
            continue
        
        window_labels = np.load(labels_path)
        
        # 查找所有窗口数据文件
        window_files = [f for f in os.listdir(subject_dir) if f.endswith("_windows.npy")]
        
        # 加载每个信号的窗口
        for window_file in window_files:
            signal_name = window_file.replace("_windows.npy", "")
            windows = np.load(os.path.join(subject_dir, window_file))
            
            # 将窗口添加到合并数据中
            if signal_name not in all_windows:
                all_windows[signal_name] = []
            
            all_windows[signal_name].append(windows)
        
        # 添加标签和受试者ID
        all_labels.append(window_labels)
        all_subjects.extend([subject_id] * len(window_labels))
    
    # 如果没有找到任何数据，返回
    if not all_windows:
        print("没有找到任何窗口数据")
        return
    
    # 合并数据
    merged_windows = {}
    for signal_name, windows_list in all_windows.items():
        merged_windows[signal_name] = np.vstack(windows_list)
    
    merged_labels = np.concatenate(all_labels)
    merged_subjects = np.array(all_subjects)
    
    # 创建合并数据的输出目录
    merged_dir = os.path.join(windowed_data_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    
    # 保存合并后的数据
    for signal_name, windows in merged_windows.items():
        np.save(os.path.join(merged_dir, f"{signal_name}_windows.npy"), windows)
    
    np.save(os.path.join(merged_dir, "window_labels.npy"), merged_labels)
    np.save(os.path.join(merged_dir, "subject_ids.npy"), merged_subjects)
    
    # 保存合并数据的信息
    with open(os.path.join(merged_dir, "merged_info.txt"), "w") as f:
        f.write("合并窗口数据信息\n")
        f.write("==============\n\n")
        f.write(f"包含的受试者: {len(subject_dirs)}\n")
        f.write(f"总窗口数: {len(merged_labels)}\n\n")
        
        # 标签分布
        unique_labels, counts = np.unique(merged_labels, return_counts=True)
        f.write("标签分布:\n")
        for label, count in zip(unique_labels, counts):
            label_name = "基线" if label == 1 else "压力" if label == 2 else f"未知 ({label})"
            f.write(f"  标签 {label} ({label_name}): {count} 窗口 ({count/len(merged_labels)*100:.2f}%)\n")
        
        # 每个受试者的窗口数量
        f.write("\n每个受试者的窗口数量:\n")
        unique_subjects, counts = np.unique(merged_subjects, return_counts=True)
        for subject, count in zip(unique_subjects, counts):
            f.write(f"  {subject}: {count} 窗口\n")
        
        # 窗口形状
        f.write("\n窗口形状:\n")
        for signal_name, windows in merged_windows.items():
            f.write(f"  {signal_name}: {windows.shape}\n")
    
    print(f"合并完成，总共 {len(merged_labels)} 个窗口，保存到 {merged_dir}")
    print(f"标签分布: {dict(zip(unique_labels, counts))}")

if __name__ == "__main__":
    # 配置参数
    input_dir = "/Volumes/xcy/TeamProject/WESAD/normalized_data"  # 标准化后数据的目录
    output_dir = "/Volumes/xcy/TeamProject/WESAD/windowed_data_60s"  # 窗口化数据的新输出目录
    window_size_sec = 60  # 窗口大小（秒）
    step_size_sec = 30  # 窗口步长（秒）
    
    print("="*80)
    print("WESAD数据窗口化处理（滑动窗口方法）")
    print(f"窗口大小: {window_size_sec}秒, 步长: {step_size_sec}秒")
    print("="*80)
    
    # 窗口化处理
    window_all_subjects(input_dir, output_dir, window_size_sec, step_size_sec)
    
    print("\n窗口化处理完成！")
    print(f"窗口化后的数据保存在 {output_dir}")
    print("="*80) 