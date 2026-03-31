"""Data loading and preprocessing for the LANL Cyber Security Dataset.

LANL auth.txt format (comma-delimited):
    time, source_user@domain, dest_user@domain, source_computer,
    dest_computer, auth_type, logon_type, auth_orientation, success/failure

Red team file (redteam.txt) format:
    time, source_user@domain, source_computer, dest_computer
"""

import os
import gzip
import pandas as pd
import numpy as np
from src.utils import RAW_DIR, PROCESSED_DIR

AUTH_COLUMNS = [
    'time', 'src_user_domain', 'dst_user_domain',
    'src_computer', 'dst_computer',
    'auth_type', 'logon_type', 'auth_orientation', 'outcome'
]

REDTEAM_COLUMNS = ['time', 'src_user_domain', 'src_computer', 'dst_computer']


def _find_auth_file():
    """Locate auth.txt or auth.txt.gz in data/raw/."""
    for name in ['auth.txt.gz', 'auth.txt']:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Place auth.txt.gz (or auth.txt) in {RAW_DIR}/ before running."
    )


def _find_redteam_file():
    """Locate redteam.txt or redteam.txt.gz in data/raw/."""
    for name in ['redteam.txt.gz', 'redteam.txt']:
        path = os.path.join(RAW_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Place redteam.txt.gz (or redteam.txt) in {RAW_DIR}/ before running."
    )


def load_auth_data(nrows: int = 1_000_000) -> pd.DataFrame:
    """Load authentication events from the LANL dataset.

    Parameters
    ----------
    nrows : int
        Number of rows to load (default 1M ≈ 1% of full dataset).
    """
    path = _find_auth_file()
    print(f"Loading auth data from {path} ({nrows:,} rows)...")

    df = pd.read_csv(
        path,
        names=AUTH_COLUMNS,
        nrows=nrows,
        na_values='?',
        low_memory=False,
    )
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Time range: {df['time'].min()} – {df['time'].max()} seconds")
    return df


def load_redteam_labels() -> pd.DataFrame:
    """Load red team (known malicious) event labels."""
    path = _find_redteam_file()
    print(f"Loading red team labels from {path}...")

    df = pd.read_csv(path, names=REDTEAM_COLUMNS, na_values='?')
    print(f"  Loaded {len(df):,} red team events")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and enrich raw authentication data.

    - Split user@domain into separate user and domain columns
    - Convert outcome to binary (1 = success, 0 = failure)
    - Add basic time features
    """
    df = df.copy()

    # Parse source user@domain
    src_split = df['src_user_domain'].str.split('@', n=1, expand=True)
    df['src_user'] = src_split[0]
    df['src_domain'] = src_split[1] if src_split.shape[1] > 1 else np.nan

    # Parse destination user@domain
    dst_split = df['dst_user_domain'].str.split('@', n=1, expand=True)
    df['dst_user'] = dst_split[0]
    df['dst_domain'] = dst_split[1] if dst_split.shape[1] > 1 else np.nan

    # Binary outcome
    df['success'] = (df['outcome'] == 'Success').astype(int)

    # Time features (LANL time is in seconds from epoch=1)
    df['hour'] = (df['time'] % 86400) // 3600          # hour of day (0-23)
    df['day'] = df['time'] // 86400                     # day number
    df['day_of_week'] = df['day'] % 7                   # 0=Mon … 6=Sun
    df['is_off_hours'] = ((df['hour'] < 6) | (df['hour'] >= 22)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    print(f"  Cleaned data: {len(df):,} rows, {df.shape[1]} columns")
    return df


def label_redteam_events(df: pd.DataFrame, redteam: pd.DataFrame) -> pd.DataFrame:
    """Add a binary 'is_suspicious' column by matching red team events.

    Matching is done on (time, src_user_domain, src_computer, dst_computer).
    """
    df = df.copy()

    redteam_keys = set(
        zip(redteam['time'], redteam['src_user_domain'],
            redteam['src_computer'], redteam['dst_computer'])
    )

    df['is_suspicious'] = df.apply(
        lambda r: 1 if (r['time'], r['src_user_domain'],
                        r['src_computer'], r['dst_computer']) in redteam_keys
        else 0, axis=1
    )

    n_sus = df['is_suspicious'].sum()
    print(f"  Labeled {n_sus:,} suspicious events "
          f"({n_sus/len(df)*100:.4f}% of total)")
    return df


def save_processed_data(df: pd.DataFrame, filename: str):
    """Save a DataFrame to the processed data directory as parquet."""
    path = os.path.join(PROCESSED_DIR, filename)
    df.to_parquet(path, index=False)
    print(f"  Saved {len(df):,} rows to {path}")
