#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Performs Leave-One-Subject-Out (LOSO) cross-validation on your windowed WESAD data,
training a 1D-CNN on each fold and reporting per-subject accuracies and the overall mean/std.
"""

import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow as tf

def build_model(input_shape, num_classes=2):
    """Builds and returns the 1D-CNN model."""
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        # First conv block
        tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        # Second conv block
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        # Global pooling
        tf.keras.layers.GlobalAveragePooling1D(),
        # Classification head
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    # ------------------ CONFIGURATION ------------------ #
    windowed_dir = "/kaggle/working/WESAD/windowed_wrist_data"
    merged_dir   = os.path.join(windowed_dir, "merged")
    # ---------------------------------------------------- #

    # 0) GPU report
    print("GPUs:", tf.config.list_physical_devices('GPU'))

    # 1) Load merged windows and labels
    X = np.load(os.path.join(merged_dir, "X_windows_all.npy"))
    y = np.load(os.path.join(merged_dir, "y_windows_all.npy"))

    # 2) Build groups array (one entry per window)
    subject_dirs = sorted([
        d for d in os.listdir(windowed_dir)
        if os.path.isdir(os.path.join(windowed_dir, d)) and d != "merged"
    ])
    groups = []
    for sid in subject_dirs:
        subj_X = np.load(os.path.join(windowed_dir, sid, "X_windows.npy"))
        groups.extend([sid] * subj_X.shape[0])
    groups = np.array(groups)
    # Map subject IDs to integers
    unique_subjs = sorted(set(groups))
    subj_to_int  = {s:i for i,s in enumerate(unique_subjs)}
    groups_int   = np.array([subj_to_int[s] for s in groups])

    # 3) Initialize LOSO splitter
    logo = LeaveOneGroupOut()

    # 4) LOSO training loop
    accuracies = []
    for fold, (train_idx, test_idx) in enumerate(
            logo.split(X, y, groups_int), start=1):

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 5) Normalize per-channel on train set
        mean = X_train.mean(axis=(0,1), keepdims=True)
        std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test  = (X_test  - mean) / std

        # 6) One-hot encode labels (assuming y in {1,2})
        y_train_cat = tf.keras.utils.to_categorical(y_train - 1, num_classes=2)
        y_test_cat  = tf.keras.utils.to_categorical(y_test  - 1, num_classes=2)

        # 7) Build & train model
        model = build_model(input_shape=X_train.shape[1:], num_classes=2)
        es = tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True)
        model.fit(
            X_train, y_train_cat,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            callbacks=[es],
            verbose=0
        )

        # 8) Evaluate on held-out subject
        loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
        subj = unique_subjs[groups_int[test_idx][0]]
        print(f"Fold {fold} — Test subject {subj}: accuracy = {acc:.3f}")
        accuracies.append(acc)

    # 9) Report overall LOSO results
    mean_acc = np.mean(accuracies)
    std_acc  = np.std(accuracies)
    print(f"\nLOSO mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

if __name__ == "__main__":
    main()
