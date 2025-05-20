#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('pip install pyts')

# Importing the needed libraries 
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision import models
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from scipy.signal import resample
from pyts.image import GramianAngularField

# Clearly defining the device to make sure that pytorch is using the gpu at hand 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Currently using device: {device}")


# Defining the dataset class to ease up the preprocessing steps 
class WESADWristGAFDataset(Dataset):
    def __init__(self, pkl_paths, window_s=60, stride_s=2, transform=None, img_size=224):
        self.transform = transform
        self.window_s = window_s
        self.stride_s = stride_s
        self.img_size = img_size
        self.resampled = {}      # maps pkl -> (3xN array, labels)
        self.index_map = []      # list of (pkl, start_idx)

        # Preparing the GAF encoder
        self.gaf = GramianAngularField(image_size=img_size, method='summation')
        #self.gaf_diff=GramianAngularField(image_size=ing_size, method='difference')

        for pkl in pkl_paths:
            with open(pkl, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            labels = np.array(data['label'])  # 700Hz labels
            N = len(labels)
            # The raw wrist streams
            wrist = data['signal']['wrist']
            acc_raw = np.array(wrist['ACC'])      # (N_acc,3)
            bvp_raw = np.array(wrist['BVP']).ravel()  # (N_bvp,)
            eda_raw = np.array(wrist['EDA']).ravel()  # (N_eda,)
            # Resampling all to N (700Hz)
            acc_r = resample(acc_raw, N, axis=0)
            acc_mag = np.linalg.norm(acc_r, axis=1)
            bvp_r = resample(bvp_raw, N)
            eda_r = resample(eda_raw, N)
            arr3 = np.stack((acc_mag, bvp_r, eda_r), axis=0)  # (3,N)
            self.resampled[pkl] = (arr3, labels)
            # Sliding windows
            win = int(window_s * 700)
            step = int(stride_s * 700)
            for start in range(0, N - win + 1, step):
                major = np.bincount(labels[start:start+win]).argmax()
                if major in [1,2,3,4]:
                    self.index_map.append((pkl, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        pkl, start = self.index_map[idx]
        arr3, labels = self.resampled[pkl]  # arr3: (3,N)
        win = int(self.window_s * 700)
        slice3 = arr3[:, start:start+win]  # (3, win)
        # binary label
        major = np.bincount(labels[start:start+win]).argmax()
        label = 1 if major == 2 else 0
        # Downsampling slice to img_size length
        slice_ds = resample(slice3, self.img_size, axis=1)  # (3, img_size)
        # GAF encoding per channel -> list of (img_size, img_size)
        gaf_imgs = self.gaf.fit_transform(slice_ds)
        # Stacking into tensor (3, img_size, img_size)
        img = torch.tensor(gaf_imgs, dtype=torch.float)
        if self.transform:
            img = self.transform(img)
        return img, label


data_root = '/kaggle/input/wesaddataset/WESAD'
pkl_paths = [os.path.join(rt, f)
             for rt, _, fls in os.walk(data_root)
             for f in fls if f.endswith('.pkl')]
# Full dataset
dataset = WESADWristGAFDataset(pkl_paths, window_s=60, stride_s=2, transform=None, img_size=224)
# extract subject ids for each window
groups = [int(os.path.basename(os.path.dirname(p))[1:]) for p,_ in dataset.index_map]

# Splitting subjects: 20% for testing, 20% validation from training set
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
trainval_idx, test_idx = next(gss_test.split(range(len(dataset)), groups=groups, y=[0]*len(dataset)))
# second split
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx_rel, val_idx_rel = next(gss_val.split(trainval_idx,
                                           groups=[groups[i] for i in trainval_idx],
                                           y=[0]*len(trainval_idx)))
train_idx = [trainval_idx[i] for i in train_idx_rel]
val_idx   = [trainval_idx[i] for i in val_idx_rel]

# subsets
train_ds = Subset(dataset, train_idx)
val_ds   = Subset(dataset, val_idx)
test_ds  = Subset(dataset, test_idx)

# transforms
train_tf = T.Compose([
    T.RandomResizedCrop((128,128), scale=(0.8,1.0)),
    T.RandomHorizontalFlip(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
testval_tf = T.Compose([
    T.Resize((128,128)),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])
class Wrapped(Dataset):
    def __init__(self, ds, tf): self.ds=ds; self.tf=tf
    def __len__(self): return len(self.ds)
    def __getitem__(self,i): img,lbl=self.ds[i]; return self.tf(img), lbl

train_loader = DataLoader(Wrapped(train_ds, train_tf), batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(Wrapped(val_ds,   testval_tf), batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(Wrapped(test_ds,  testval_tf), batch_size=32, shuffle=False, num_workers=4)

print(f"Subjects: total={len(set(groups))}, train={len({groups[i] for i in train_idx})}, "
      f"val={len({groups[i] for i in val_idx})}, test={len({groups[i] for i in test_idx})}")


model = models.resnet18(pretrained=True)
in_feat = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_feat, 2)
)
model = model.to(device)
# freeze
for name,param in model.named_parameters():
    if not name.startswith('layer4') and 'fc' not in name:
        param.requires_grad=False

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)




def train_one_epoch(model, loader, crit, opt, dev):
    model.train(); loss_sum=correct=total=0
    for X,y in tqdm(loader, desc='Train'):
        X,y=X.to(dev),y.to(dev)
        opt.zero_grad(); logits=model(X); loss=crit(logits,y)
        loss.backward(); opt.step()
        loss_sum+=loss.item()*X.size(0)
        preds=logits.argmax(1); correct+=(preds==y).sum().item(); total+=y.size(0)
    return loss_sum/total, correct/total

def validate(model, loader, crit, dev):
    model.eval(); loss_sum=correct=total=0
    with torch.no_grad():
        for X,y in tqdm(loader, desc='Val  '):
            X,y=X.to(dev),y.to(dev)
            logits=model(X); loss=crit(logits,y)
            loss_sum+=loss.item()*X.size(0)
            preds=logits.argmax(1); correct+=(preds==y).sum().item(); total+=y.size(0)
    return loss_sum/total, correct/total

epochs=30; best_val_loss=float('inf'); patience=5; counter=0; ckpt='/kaggle/working/best_wrist_resnet18.pth'
for e in range(1, epochs+1):
    tr_l, tr_a = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_l, val_a = validate(model, val_loader, criterion, device)
    scheduler.step()
    print(f"Epoch {e}/{epochs} | Train Acc: {tr_a*100:.2f}% | Val Acc: {val_a*100:.2f}% | Val Loss: {val_l:.4f}")
    if val_l<best_val_loss:
        best_val_loss=val_l; counter=0; torch.save(model.state_dict(), ckpt); print(" New best model")
    else:
        counter+=1
        if counter>=patience: print(f"Early stopping at epoch {e}"); break


model.load_state_dict(torch.load(ckpt)); model.eval()
test_l, test_a = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_l:.4f}, Test Acc: {test_a*100:.2f}%")





