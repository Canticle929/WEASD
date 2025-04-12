import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set dataset path
dataset_path = "/Volumes/xcy/Team Project/WESAD"

# Load dataset
def load_data(path):
    data = {}
    for subject_dir in os.listdir(path):
        if subject_dir.startswith('S'):
            subject_path = os.path.join(path, subject_dir)
            pickle_file = os.path.join(subject_path, subject_dir + '.pkl')
            
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as f:
                    data[subject_dir] = pickle.load(f, encoding='latin1')
    
    return data

print("Starting data loading...")
data = load_data(dataset_path)
print(f"Data loaded. Found {len(data)} subjects.")

# Explore data structure
subject = list(data.keys())[0]  # Select the first subject
print("Exploring data structure...")
print(f"Available subjects: {list(data.keys())}")
print(f"Data structure: {data[subject].keys()}")
print(f"Signal types: {data[subject]['signal'].keys()}")
print(f"Labels: {np.unique(data[subject]['label'])}")

# Visualize some signals
plt.figure(figsize=(15, 10))

# Select signals for visualization (e.g., ECG, EDA, BVP)
signals = ['chest', 'wrist']
for i, device in enumerate(signals):
    if device in data[subject]['signal']:
        for j, signal_name in enumerate(['ACC', 'ECG', 'EDA', 'TEMP']):
            if signal_name in data[subject]['signal'][device]:
                plt.subplot(len(signals), 4, i*4 + j + 1)
                signal = data[subject]['signal'][device][signal_name]
                if len(signal.shape) > 1:
                    signal = signal[:, 0]  # If multi-dimensional, take only the first dimension
                plt.plot(signal[:1000])  # Only show the first 1000 sample points
                plt.title(f"{device} - {signal_name}")

plt.tight_layout()
plt.show()

print("\nDetailed data structure for chest signals:")
if 'chest' in data[subject]['signal']:
    print(f"Chest signal keys: {data[subject]['signal']['chest'].keys()}")
    
    # 检查是否有采样率信息
    if 'sampling_rate' not in data[subject]['signal']['chest']:
        print("'sampling_rate' key not found. Looking for alternative keys...")
        # 检查其他可能的键名
        for key in data[subject]['signal']['chest'].keys():
            if isinstance(data[subject]['signal']['chest'][key], (int, float)) and key != 'label':
                print(f"Possible sampling rate key: {key} = {data[subject]['signal']['chest'][key]}") 