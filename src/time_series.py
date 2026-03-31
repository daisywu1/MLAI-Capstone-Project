"""Time series analysis for authentication event data."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from src.utils import IMAGES_DIR


def _save_fig(name: str):
    path = os.path.join(IMAGES_DIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")


def aggregate_login_timeseries(df: pd.DataFrame,
                                freq_seconds: int = 3600) -> pd.Series:
    """Aggregate authentication events into fixed-interval time bins.

    Parameters
    ----------
    df : DataFrame with 'time' column (seconds since epoch)
    freq_seconds : bin width in seconds (default 3600 = 1 hour)

    Returns
    -------
    pd.Series indexed by bin start time, values = event count
    """
    bins = df['time'] // freq_seconds
    ts = bins.value_counts().sort_index()
    ts.index = ts.index * freq_seconds

    # Fill gaps with zeros
    full_idx = range(ts.index.min(), ts.index.max() + freq_seconds, freq_seconds)
    ts = ts.reindex(full_idx, fill_value=0)
    ts.index.name = 'time_bin'
    ts.name = 'event_count'
    return ts


def decompose_timeseries(ts: pd.Series, period: int = 24):
    """Perform seasonal decomposition on the time series.

    Parameters
    ----------
    ts : event count time series
    period : seasonality period (default 24 = daily for hourly data)

    Returns
    -------
    DecomposeResult with trend, seasonal, resid components
    """
    decomposition = seasonal_decompose(ts, model='additive', period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(ts.values, color='steelblue', linewidth=0.8)
    axes[0].set_ylabel('Observed')
    axes[0].set_title('Time Series Decomposition')

    axes[1].plot(decomposition.trend.values, color='darkorange', linewidth=1.5)
    axes[1].set_ylabel('Trend')

    axes[2].plot(decomposition.seasonal.values, color='green', linewidth=0.8)
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(decomposition.resid.values, color='red', linewidth=0.5)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Time Bin Index')

    plt.tight_layout()
    _save_fig('ts_decomposition')
    plt.show()

    return decomposition


def detect_temporal_anomalies(ts: pd.Series, window: int = 24,
                               threshold: float = 3.0) -> pd.Series:
    """Detect anomalous time bins using rolling z-score.

    Parameters
    ----------
    ts : event count time series
    window : rolling window size (default 24 = 1 day for hourly data)
    threshold : z-score threshold for anomaly (default 3.0)

    Returns
    -------
    Boolean Series where True = anomalous bin
    """
    rolling_mean = ts.rolling(window=window, center=True).mean()
    rolling_std = ts.rolling(window=window, center=True).std()

    z_scores = (ts - rolling_mean) / rolling_std.replace(0, np.nan)
    anomalies = z_scores.abs() > threshold

    n_anomalies = anomalies.sum()
    print(f"  Detected {n_anomalies} anomalous time bins "
          f"(threshold={threshold}, window={window})")
    return anomalies


def plot_timeseries_with_anomalies(ts: pd.Series, anomalies: pd.Series):
    """Plot the time series with anomalous bins highlighted."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(ts.index, ts.values, color='steelblue', linewidth=0.8,
            alpha=0.8, label='Event count')

    anom_idx = ts.index[anomalies.fillna(False)]
    anom_vals = ts[anomalies.fillna(False)]
    ax.scatter(anom_idx, anom_vals, color='red', s=30, zorder=5,
               label=f'Anomalies (n={len(anom_idx)})')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Events per Hour')
    ax.set_title('Hourly Event Volume with Temporal Anomalies')
    ax.legend()
    plt.tight_layout()
    _save_fig('ts_anomalies')
    plt.show()
