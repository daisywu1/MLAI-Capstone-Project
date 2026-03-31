"""Exploratory Data Analysis visualizations for authentication data."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import IMAGES_DIR

sns.set_style('whitegrid')
FIGSIZE = (12, 5)


def _save_fig(name: str):
    """Save the current figure to images/."""
    path = os.path.join(IMAGES_DIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")


def generate_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for the cleaned dataset."""
    print(f"Dataset shape: {df.shape}")
    print(f"\nTime range: {df['time'].min()} – {df['time'].max()} seconds")
    print(f"Duration: {(df['time'].max() - df['time'].min()) / 86400:.1f} days")
    print(f"\nUnique source users: {df['src_user'].nunique():,}")
    print(f"Unique destination users: {df['dst_user'].nunique():,}")
    print(f"Unique source computers: {df['src_computer'].nunique():,}")
    print(f"Unique destination computers: {df['dst_computer'].nunique():,}")
    print(f"\nAuthentication types: {df['auth_type'].value_counts().to_dict()}")
    print(f"Logon types: {df['logon_type'].value_counts().to_dict()}")
    print(f"\nOutcome distribution:")
    print(df['outcome'].value_counts())
    if 'is_suspicious' in df.columns:
        n_sus = df['is_suspicious'].sum()
        print(f"\nRed team events: {n_sus:,} ({n_sus/len(df)*100:.4f}%)")


def plot_login_distribution(df: pd.DataFrame):
    """Plot login volume over time (by day)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    daily = df.groupby('day').size()
    ax.bar(daily.index, daily.values, color='steelblue', alpha=0.8)
    ax.set_xlabel('Day')
    ax.set_ylabel('Number of Events')
    ax.set_title('Daily Authentication Event Volume')
    plt.tight_layout()
    _save_fig('eda_daily_volume')
    plt.show()


def plot_success_failure_ratio(df: pd.DataFrame):
    """Plot success vs. failure breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

    counts = df['outcome'].value_counts()
    axes[0].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0].set_title('Success vs. Failure')

    daily_fail = df.groupby('day')['success'].apply(lambda x: 1 - x.mean())
    axes[1].plot(daily_fail.index, daily_fail.values, color='#e74c3c', linewidth=1.5)
    axes[1].set_xlabel('Day')
    axes[1].set_ylabel('Failure Rate')
    axes[1].set_title('Daily Failure Rate Over Time')

    plt.tight_layout()
    _save_fig('eda_success_failure')
    plt.show()


def plot_hourly_pattern(df: pd.DataFrame):
    """Plot hourly login pattern."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    hourly = df.groupby('hour').size()
    ax.bar(hourly.index, hourly.values, color='darkorange', alpha=0.8)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Events')
    ax.set_title('Hourly Authentication Pattern')
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    _save_fig('eda_hourly_pattern')
    plt.show()


def plot_top_users(df: pd.DataFrame, top_n: int = 20):
    """Plot most active source users."""
    fig, ax = plt.subplots(figsize=(12, 6))
    top = df['src_user'].value_counts().head(top_n)
    ax.barh(range(len(top)), top.values, color='teal', alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel('Number of Events')
    ax.set_title(f'Top {top_n} Most Active Source Users')
    ax.invert_yaxis()
    plt.tight_layout()
    _save_fig('eda_top_users')
    plt.show()


def plot_auth_type_distribution(df: pd.DataFrame):
    """Plot authentication type distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df['auth_type'].value_counts()
    ax.bar(counts.index, counts.values, color='mediumpurple', alpha=0.8)
    ax.set_xlabel('Authentication Type')
    ax.set_ylabel('Count')
    ax.set_title('Authentication Type Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _save_fig('eda_auth_types')
    plt.show()


def plot_logon_type_distribution(df: pd.DataFrame):
    """Plot logon type distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df['logon_type'].value_counts()
    ax.bar(counts.index, counts.values, color='coral', alpha=0.8)
    ax.set_xlabel('Logon Type')
    ax.set_ylabel('Count')
    ax.set_title('Logon Type Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    _save_fig('eda_logon_types')
    plt.show()


def plot_redteam_timeline(df: pd.DataFrame):
    """Plot red team events over time."""
    if 'is_suspicious' not in df.columns or df['is_suspicious'].sum() == 0:
        print("  No red team events to plot.")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sus = df[df['is_suspicious'] == 1]
    daily_sus = sus.groupby('day').size()
    daily_total = df.groupby('day').size()

    ax.bar(daily_total.index, daily_total.values, color='steelblue',
           alpha=0.4, label='All events')
    ax2 = ax.twinx()
    ax2.bar(daily_sus.index, daily_sus.values, color='red',
            alpha=0.8, label='Red team events', width=0.4)
    ax.set_xlabel('Day')
    ax.set_ylabel('Total Events', color='steelblue')
    ax2.set_ylabel('Red Team Events', color='red')
    ax.set_title('Red Team Events Over Time')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    _save_fig('eda_redteam_timeline')
    plt.show()


def plot_correlation_matrix(df_features: pd.DataFrame):
    """Plot correlation heatmap of engineered features."""
    from src.feature_engineering import TARGET_COLUMN
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()

    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df_features[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, ax=ax, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    _save_fig('eda_correlation_matrix')
    plt.show()


def run_full_eda(df: pd.DataFrame):
    """Run all EDA visualizations."""
    generate_summary_statistics(df)
    plot_login_distribution(df)
    plot_success_failure_ratio(df)
    plot_hourly_pattern(df)
    plot_top_users(df)
    plot_auth_type_distribution(df)
    plot_logon_type_distribution(df)
    plot_redteam_timeline(df)
