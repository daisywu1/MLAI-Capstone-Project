"""PCA dimensionality reduction and K-Means clustering."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.utils import IMAGES_DIR


def _save_fig(name: str):
    path = os.path.join(IMAGES_DIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")


def plot_explained_variance(X: np.ndarray, max_components: int = None):
    """Plot cumulative explained variance ratio for PCA."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n = max_components or min(X_scaled.shape[1], 15)
    pca = PCA(n_components=n)
    pca.fit(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(1, n + 1), pca.explained_variance_ratio_,
                color='steelblue', alpha=0.8)
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Individual Explained Variance')

    axes[1].plot(range(1, n + 1), cumvar, 'o-', color='darkorange', linewidth=2)
    axes[1].axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend()

    plt.tight_layout()
    _save_fig('pca_explained_variance')
    plt.show()

    for i, (var, cum) in enumerate(zip(pca.explained_variance_ratio_, cumvar)):
        print(f"  PC{i+1}: {var:.4f} (cumulative: {cum:.4f})")


def apply_pca(X: np.ndarray, n_components: int = 2):
    """Apply PCA and return transformed data, fitted PCA, and scaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"  PCA: {X.shape[1]} features → {n_components} components")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return X_pca, pca, scaler


def find_optimal_clusters(X_pca: np.ndarray, max_k: int = 8):
    """Elbow method and silhouette analysis for optimal K."""
    K_range = range(2, max_k + 1)
    inertias = []
    silhouettes = []

    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca, labels, sample_size=min(10000, len(X_pca)))
        silhouettes.append(sil)
        print(f"  K={k}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(K_range), inertias, 'o-', color='steelblue', linewidth=2)
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')

    axes[1].plot(list(K_range), silhouettes, 'o-', color='darkorange', linewidth=2)
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')

    plt.tight_layout()
    _save_fig('clustering_elbow_silhouette')
    plt.show()

    return list(K_range), inertias, silhouettes


def apply_kmeans(X_pca: np.ndarray, n_clusters: int = 3):
    """Apply K-Means clustering."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=min(10000, len(X_pca)))
    print(f"  K-Means (K={n_clusters}): silhouette={sil:.4f}")
    return labels, km


def plot_pca_clusters(X_pca: np.ndarray, labels: np.ndarray,
                      suspicious: np.ndarray = None):
    """Visualize PCA clusters with optional red team overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                              cmap='viridis', alpha=0.3, s=5)
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('K-Means Clusters in PCA Space')
    plt.colorbar(scatter, ax=axes[0], label='Cluster')

    if suspicious is not None:
        normal = suspicious == 0
        sus = suspicious == 1
        axes[1].scatter(X_pca[normal, 0], X_pca[normal, 1],
                        c='steelblue', alpha=0.2, s=3, label='Normal')
        axes[1].scatter(X_pca[sus, 0], X_pca[sus, 1],
                        c='red', alpha=0.8, s=20, label='Red Team', zorder=5)
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('PCA Space: Normal vs. Red Team')
        axes[1].legend()
    else:
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                        cmap='viridis', alpha=0.3, s=5)
        axes[1].set_title('Cluster Assignment')

    plt.tight_layout()
    _save_fig('pca_clusters')
    plt.show()
