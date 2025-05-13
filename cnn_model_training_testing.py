# --------------- Splitting data and  model training and testing ---------------

import os
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
import tensorflow as tf

print("GPUs:", tf.config.list_physical_devices('GPU'))

# —————————————— CONFIG ————————————————— #
windowed_dir = "/kaggle/working/WESAD/windowed_wrist_data"
merged_dir   = os.path.join(windowed_dir, "merged")
# ———————————————————————————————— #

# 1) Load merged windows and labels
X = np.load(os.path.join(merged_dir, "X_windows_all.npy"))    # (n_windows, win_len, channels)
y = np.load(os.path.join(merged_dir, "y_windows_all.npy"))    # (n_windows,)

# 2) Build groups array
groups = []
subject_dirs = sorted([
    d for d in os.listdir(windowed_dir)
    if os.path.isdir(os.path.join(windowed_dir, d)) and d != "merged"
])
for sid in subject_dirs:
    subj_X = np.load(os.path.join(windowed_dir, sid, "X_windows.npy"))
    n_win = subj_X.shape[0]
    groups.extend([sid] * n_win)
groups = np.array(groups)
# map string IDs to integers
unique_subjs = sorted(set(groups))
subj_to_int = {s: i for i, s in enumerate(unique_subjs)}
groups_int = np.array([subj_to_int[s] for s in groups])

# 3) Initialize LOSO
logo = LeaveOneGroupOut()

# 4) Loop over folds
accuracies = []
for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups_int), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 5) Normalize per-channel on train only
    mean = X_train.mean(axis=(0,1), keepdims=True)
    std  = X_train.std(axis=(0,1), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    # 6) One-hot encode labels (assumes labels are 1 or 2)
    y_train_cat = tf.keras.utils.to_categorical(y_train - 1, num_classes=2)
    y_test_cat  = tf.keras.utils.to_categorical(y_test  - 1, num_classes=2)

    # 7) Build a simple 1D-CNN


    model = tf.keras.Sequential([
        # 1) Explicit Input layer—best practice  
        tf.keras.Input(shape=X_train.shape[1:]),  # (window_length, n_channels)
    
        # 2) First conv block: larger kernel for mid-range patterns
        tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
    
        # 3) Second conv block: deeper, finer patterns
        tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
    
        # 4) Global pooling to collapse time-axis without huge parameter blow-up
        tf.keras.layers.GlobalAveragePooling1D(),
    
        # 5) Classification head
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


    # 8) Train with early stopping
    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_train, y_train_cat,
              epochs=50,
              batch_size=32,
              validation_split=0.1,
              callbacks=[es],
              verbose=0)

    # 9) Evaluate on the held‐out subject
    loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Fold {fold} — Test subject {unique_subjs[groups_int[test_idx][0]]}: accuracy = {acc:.3f}")
    accuracies.append(acc)

# 10) Aggregate results
mean_acc = np.mean(accuracies)
std_acc  = np.std(accuracies)
print(f"\nLOSO mean accuracy: {mean_acc:.3f} ± {std_acc:.3f}")


