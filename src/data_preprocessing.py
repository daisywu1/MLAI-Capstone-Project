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

# ~1.6 billion total rows in the full dataset
_TOTAL_ROWS_ESTIMATE = 1_648_275_307


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
    """Load the first N authentication events (fast, but covers limited time).

    Parameters
    ----------
    nrows : int
        Number of rows to load from the start of the file.
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


def load_auth_data_sampled(target_rows: int = 1_000_000,
                           chunk_size: int = 5_000_000,
                           random_state: int = 42) -> pd.DataFrame:
    """Load a sample of authentication events spread across the full dataset.

    Reads the file in chunks and randomly samples from each chunk so the
    result covers the entire 58-day time range. It explicitly hunts for and
    retains 100% of the known red team events it encounters to ensure the
    resulting sample has positive labels for model training.

    Parameters
    ----------
    target_rows : int
        Approximate number of rows in the final sample.
    chunk_size : int
        Rows per chunk when reading the file.
    random_state : int
        Random seed for reproducibility.
    """
    path = _find_auth_file()
    sampling_rate = target_rows / _TOTAL_ROWS_ESTIMATE
    print(f"Loading stratified sample (~{target_rows:,} rows, "
          f"rate={sampling_rate:.6f}) from {path}...")

    rng = np.random.RandomState(random_state)
    chunks = []
    rows_collected = 0

    # Pre-load red team labels to ensure we don't drop them during sampling
    redteam = load_redteam_labels()
    redteam_keys = set(
        zip(redteam['time'], redteam['src_user_domain'],
            redteam['src_computer'], redteam['dst_computer'])
    )

    reader = pd.read_csv(
        path,
        names=AUTH_COLUMNS,
        na_values='?',
        low_memory=False,
        chunksize=chunk_size,
    )

    for i, chunk in enumerate(reader):
        # 1. Identify and keep all red team events in this chunk
        is_red = chunk.apply(
            lambda r: (r['time'], r['src_user_domain'],
                       r['src_computer'], r['dst_computer']) in redteam_keys,
            axis=1
        )
        red_chunk = chunk[is_red]

        # 2. Sample from the normal events
        normal_chunk = chunk[~is_red]
        n_sample = max(1, int(len(normal_chunk) * sampling_rate))
        sampled_normal = normal_chunk.sample(n=min(n_sample, len(normal_chunk)), random_state=rng)

        # Combine and store
        combined = pd.concat([red_chunk, sampled_normal])
        chunks.append(combined)
        rows_collected += len(combined)

        if (i + 1) % 20 == 0:
            print(f"  ... processed {(i+1)*chunk_size/1e6:.0f}M rows, "
                  f"collected {rows_collected:,} so far")

        if rows_collected >= target_rows * 1.1:
            break

    df = pd.concat(chunks, ignore_index=True).sort_values('time').reset_index(drop=True)

    # If we overshot the target, downsample the normal events again, keeping all red team events
    if len(df) > target_rows:
        is_red_final = df.apply(
            lambda r: (r['time'], r['src_user_domain'],
                       r['src_computer'], r['dst_computer']) in redteam_keys,
            axis=1
        )
        df_red = df[is_red_final]
        df_normal = df[~is_red_final]

        needed_normal = target_rows - len(df_red)
        if needed_normal > 0:
            df_normal_sampled = df_normal.sample(n=needed_normal, random_state=random_state)
            df = pd.concat([df_red, df_normal_sampled]).sort_values('time').reset_index(drop=True)
        else:
            df = df_red

    days = (df['time'].max() - df['time'].min()) / 86400
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Time range: {df['time'].min()} – {df['time'].max()} seconds "
          f"({days:.1f} days)")
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
