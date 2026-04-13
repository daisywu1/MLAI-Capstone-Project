"""Feature engineering for authentication anomaly detection.

Builds a per-event feature matrix from cleaned LANL authentication data.
"""

import pandas as pd
import numpy as np

TARGET_COLUMN = 'is_suspicious'

FEATURE_COLUMNS = [
    'hour', 'day_of_week', 'is_off_hours', 'is_weekend',
    'success',
    'login_count_1h', 'login_count_6h', 'login_count_24h',
    'fail_count_1h', 'fail_ratio_24h',
    'unique_dst_computers_24h', 'unique_auth_types_24h', 'unique_logon_types_24h',
    'time_since_last_login',
    'is_new_dst_computer',
    'events_per_minute_10m',
]


def _rolling_counts(df: pd.DataFrame, user_col: str, time_col: str,
                    window_seconds: int) -> pd.Series:
    """Count events per user within a rolling time window."""
    counts = []
    grouped = df.groupby(user_col)[time_col].apply(list).to_dict()

    time_vals = df[time_col].values
    user_vals = df[user_col].values

    user_times = {}
    for u, times in grouped.items():
        arr = np.array(sorted(times))
        user_times[u] = arr

    result = np.zeros(len(df), dtype=np.int32)
    for idx in range(len(df)):
        u = user_vals[idx]
        t = time_vals[idx]
        arr = user_times[u]
        result[idx] = int(np.sum((arr >= t - window_seconds) & (arr <= t)))

    return pd.Series(result, index=df.index)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix from cleaned authentication data.

    This is the most compute-intensive step. For large datasets, consider
    processing in chunks.
    """
    df = df.copy().sort_values('time').reset_index(drop=True)

    print("  Computing rolling login counts (1h, 6h, 24h)...")
    df['login_count_1h'] = _rolling_counts(df, 'src_user', 'time', 3600)
    df['login_count_6h'] = _rolling_counts(df, 'src_user', 'time', 21600)
    df['login_count_24h'] = _rolling_counts(df, 'src_user', 'time', 86400)

    print("  Computing failure counts and ratios...")
    df_fail = df[df['success'] == 0]
    df['fail_count_1h'] = _rolling_counts(
        df.assign(_is_fail=1 - df['success']),
        'src_user', 'time', 3600
    ) * (1 - df['success'])
    # Simpler approach: ratio of failures in 24h window
    df['fail_ratio_24h'] = np.where(
        df['login_count_24h'] > 0,
        1 - (df['login_count_24h'] - df['fail_count_1h']) / df['login_count_24h'],
        0
    ).clip(0, 1)

    print("  Computing diversity features (unique computers, auth types)...")
    # For each event, count distinct destinations in the past 24h for that user
    user_day_dst = df.groupby(['src_user', 'day'])['dst_computer'].transform('nunique')
    df['unique_dst_computers_24h'] = user_day_dst

    user_day_auth = df.groupby(['src_user', 'day'])['auth_type'].transform('nunique')
    df['unique_auth_types_24h'] = user_day_auth

    user_day_logon = df.groupby(['src_user', 'day'])['logon_type'].transform('nunique')
    df['unique_logon_types_24h'] = user_day_logon

    print("  Computing time-since-last-login...")
    df['prev_time'] = df.groupby('src_user')['time'].shift(1)
    df['time_since_last_login'] = (df['time'] - df['prev_time']).fillna(0).clip(lower=0, upper=604800)

    print("  Computing new-destination flag...")
    seen_pairs = set()
    is_new = np.zeros(len(df), dtype=np.int32)
    for idx, row in df[['src_user', 'dst_computer']].iterrows():
        pair = (row['src_user'], row['dst_computer'])
        if pair not in seen_pairs:
            is_new[idx] = 1
            seen_pairs.add(pair)
    df['is_new_dst_computer'] = is_new

    print("  Computing velocity (events per minute, 10-min window)...")
    count_10m = _rolling_counts(df, 'src_user', 'time', 600)
    df['events_per_minute_10m'] = count_10m / 10.0

    # Select final feature columns + target
    keep_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    available = [c for c in keep_cols if c in df.columns]
    missing = set(keep_cols) - set(available)
    if missing:
        print(f"  Warning: missing columns {missing}, filling with 0")
        for c in missing:
            df[c] = 0

    result = df[keep_cols].copy()
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  Feature matrix: {result.shape[0]:,} rows × {result.shape[1]} columns")
    print(f"  Suspicious events: {result[TARGET_COLUMN].sum():,} "
          f"({result[TARGET_COLUMN].mean()*100:.4f}%)")
    return result
