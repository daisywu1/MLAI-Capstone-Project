"""Shared utility functions for the capstone project."""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

for d in [RAW_DIR, PROCESSED_DIR, IMAGES_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def load_processed_data(filename: str) -> pd.DataFrame:
    """Load a previously saved processed dataset."""
    path = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed file not found: {path}")
    return pd.read_parquet(path)


def prepare_train_test(df_features: pd.DataFrame, test_size=0.2, random_state=42, apply_smote=True):
    """Split feature matrix into train/test sets with scaling and optional SMOTE.

    Returns (X_train, X_test, y_train, y_test, scaler, feature_names).
    """
    from src.feature_engineering import TARGET_COLUMN

    feature_cols = [c for c in df_features.columns if c != TARGET_COLUMN]
    X = df_features[feature_cols].values
    y = df_features[TARGET_COLUMN].values

    n_positive = int(y.sum())
    n_classes = len(np.unique(y))
    use_stratify = n_classes >= 2 and n_positive >= 2

    if not use_stratify:
        print(f"  Warning: only {n_classes} class(es) with {n_positive} positive "
              f"samples — using non-stratified split.")
    else:
        print(f"  Using stratified split with {n_classes} classes and {n_positive} positive samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if use_stratify else None
    )

    print(f"  Train: {len(X_train):,} rows ({int(y_train.sum()):,} positive)")
    print(f"  Test:  {len(X_test):,} rows ({int(y_test.sum()):,} positive)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Clip extreme scaled values to prevent overflow
    X_train = np.clip(X_train, -10, 10)
    X_test = np.clip(X_test, -10, 10)

    if apply_smote and use_stratify:
        print("  Applying SMOTE to balance the training data...")
        n_pos_train = int(y_train.sum())
        k_neighbors = min(5, n_pos_train - 1)
        if k_neighbors > 0:
            smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  After SMOTE: Train has {len(X_train):,} rows ({int(y_train.sum()):,} positive)")
        else:
            print("  NO SMOTE: Warning: Not enough positive samples in train set for SMOTE. Skipping.")

    return X_train, X_test, y_train, y_test, scaler, feature_cols
