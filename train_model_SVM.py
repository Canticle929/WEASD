#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Support Vector Machine (SVM) model on extracted WESAD features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_svm(feature_file, output_dir):
    # Load feature dataset
    df = pd.read_csv(feature_file)

    # 随机划分样本各项结果可能存在过拟合
    # X = df.drop(['label', 'subject_id'], axis=1)
    # y = df['label']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    # 正确划分方式（按 subject_id）
    unique_subjects = df['subject_id'].unique()
    train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)

    # 将 subject_id 在 train_subjects 和 test_subjects 中的样本分别划分为训练集和测试集
    train_df = df[df['subject_id'].isin(train_subjects)]
    test_df = df[df['subject_id'].isin(test_subjects)]

    # 获取特征和标签
    X_train = train_df.drop(['label', 'subject_id'], axis=1)
    y_train = train_df['label']
    X_test = test_df.drop(['label', 'subject_id'], axis=1)
    y_test = test_df['label']

    # Create SVM pipeline (Standardization + SVM)
    svm_pipeline = make_pipeline(
        StandardScaler(),
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    )

    # Train SVM
    svm_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = svm_pipeline.predict(X_test)
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "svm_model.pkl")
    joblib.dump(svm_pipeline, model_path)
    print(f"SVM model saved to {model_path}")

if __name__ == "__main__":
    feature_file = "WESAD/features/feature_dataset.csv"
    output_dir = "WESAD/SVM"
    
    print("="*80)
    print("Training SVM on WESAD features...")
    print("="*80)
    train_svm(feature_file, output_dir)
    print("="*80)
