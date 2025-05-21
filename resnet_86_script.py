#!/usr/bin/env python3
"""
Train and evaluate a 3-layer CNN on 8x8 GAF-encoded WESAD wrist data.
Includes training/validation curves, final test metrics, F1 scores, and confusion matrix.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from scipy.signal import resample
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device = get_device()

class WESADWristGAFDataset(Dataset):
    def __init__(self, pkl_paths, window_s=5, stride_s=2, img_size=8):
        self.window_s = window_s
        self.stride_s = stride_s
        self.img_size = img_size
        self.gaf = GramianAngularField(image_size=img_size, method='summation')
        self.resampled = {}
        self.index_map = []
        for pkl in pkl_paths:
            with open(pkl, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            labels = np.array(data['label'])  # 700 Hz
            N = len(labels)
            wrist = data['signal']['wrist']
            acc_raw = np.array(wrist['ACC'])      # (N_acc,3)
            bvp_raw = np.array(wrist['BVP']).ravel()  # (N_bvp,)
            eda_raw = np.array(wrist['EDA']).ravel()  # (N_eda,)
            # Resample each to length N (700 Hz)
            acc_r = resample(acc_raw, N, axis=0)
            acc_mag = np.linalg.norm(acc_r, axis=1)
            bvp_r = resample(bvp_raw, N)
            eda_r = resample(eda_raw, N)
            arr3 = np.stack((acc_mag, bvp_r, eda_r), axis=0)
            self.resampled[pkl] = (arr3, labels)
            win = int(window_s * 700)
            step = int(stride_s * 700)
            for start in range(0, N - win + 1, step):
                major = np.bincount(labels[start:start+win]).argmax()
                if major in [1,2,3,4]:  # baseline, stress, amusement, meditation
                    self.index_map.append((pkl, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        pkl, start = self.index_map[idx]
        arr3, labels = self.resampled[pkl]
        win = int(self.window_s * 700)
        slice3 = arr3[:, start:start+win]  # (3, win)
        major = np.bincount(labels[start:start+win]).argmax()
        # map {1,2,3,4} to {0,1,2,3}
        label = major - 1
        # downsample time axis to img_size
        slice_ds = resample(slice3, self.img_size, axis=1)
        # GAF encode each channel
        gaf_imgs = self.gaf.fit_transform(slice_ds)
        img = torch.tensor(gaf_imgs, dtype=torch.float)  # (3, img_size, img_size)
        # normalize to [0,1]
        img = img / img.max()
        return img, label

class StressCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)
        # after three 2x2 pools on 8x8 -> 1x1
        self.fc1 = nn.Linear(64 * 1 * 1, 6)
        self.fc2 = nn.Linear(6, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = correct = total = 0
    for X, y in tqdm(loader, desc='Train'):  
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for X, y in tqdm(loader, desc='Test '):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            loss_sum += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


def main():
    # Data paths
    data_root = 'wesad_data'
    pkl_paths = [os.path.join(root, f)
                 for root, _, files in os.walk(data_root)
                 for f in files if f.endswith('.pkl')]

    # Dataset and split
    dataset = WESADWristGAFDataset(pkl_paths, window_s=5, stride_s=2, img_size=8)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.4,
                                           random_state=42, shuffle=True)
    train_ds = Subset(dataset, train_idx)
    test_ds  = Subset(dataset, test_idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=4)
    
    # Model & training setup
    model = StressCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Metrics storage
    train_losses, train_accs = [], []
    val_losses, val_accs     = [], []

    # Training loop
    epochs = 30
    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = validate(model, test_loader,  criterion, device)
        scheduler.step()
        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(te_loss)
        val_accs.append(te_acc)
        print(f"Epoch {epoch}/{epochs} | Train Acc: {tr_acc*100:.2f}% | Test Acc: {te_acc*100:.2f}%")

    # Plot training curves
    epochs_range = range(1, len(train_losses)+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses,   label='Test Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs_range, [a*100 for a in train_accs], label='Train Acc')
    plt.plot(epochs_range, [a*100 for a in val_accs],   label='Test Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy Curve'); plt.legend()
    plt.tight_layout(); plt.savefig('training_validation_curves.png', dpi=300)
    plt.show()

    # Final test evaluation
    final_loss, final_acc = validate(model, test_loader, criterion, device)
    print(f"Final Test Loss: {final_loss:.4f}, Final Test Acc: {final_acc*100:.2f}%")

    # Classification report & F1
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['baseline','stress','amusement','meditation'], digits=4))
    f1s = f1_score(all_labels, all_preds, average=None)
    print(f"F1 scores: {f1s}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = plt.subplot()
    cax = disp.imshow(cm, cmap='Blues')
    disp.set_xticks(np.arange(4)); disp.set_yticks(np.arange(4))
    disp.set_xticklabels(['baseline','stress','amusement','meditation'], rotation=45)
    disp.set_yticklabels(['baseline','stress','amusement','meditation'])
    plt.colorbar(cax)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='white')
    plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
