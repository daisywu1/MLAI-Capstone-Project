"""KNN and Logistic Regression classification with evaluation plots."""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, RocCurveDisplay
)
from src.utils import IMAGES_DIR


def _save_fig(name: str):
    path = os.path.join(IMAGES_DIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")


def find_optimal_k(X_train, y_train, k_range=range(1, 16), cv=5):
    """Find optimal k for KNN using cross-validation.

    Returns (best_k, scores_dict).
    """
    scores = {}
    n_positive = int(y_train.sum())
    if n_positive < cv:
        cv = max(2, n_positive)
        
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if cv >= 2 else None
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        if cv_strategy:
            try:
                cv_scores = cross_val_score(knn, X_train, y_train, cv=cv_strategy,
                                            scoring='roc_auc', n_jobs=-1)
                scores[k] = cv_scores.mean()
                print(f"  k={k}: AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            except Exception:
                scores[k] = 0.5
        else:
            scores[k] = 0.5

    if not scores or all(v == 0.5 for v in scores.values()):
        print("  Warning: CV failed. Using default k=5")
        return 5, {5: 0.5}

    best_k = max(scores, key=scores.get)
    print(f"\n  Best k = {best_k} (AUC = {scores[best_k]:.4f})")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(scores.keys()), list(scores.values()), 'o-',
            color='steelblue', linewidth=2)
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.7,
               label=f'Best k = {best_k}')
    ax.set_xlabel('k (Number of Neighbors)')
    ax.set_ylabel('ROC AUC (CV)')
    ax.set_title('KNN: Optimal k Selection')
    ax.legend()
    plt.tight_layout()
    _save_fig('knn_optimal_k')
    plt.show()

    return best_k, scores


def train_knn(X_train, y_train, n_neighbors=5):
    """Train a KNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    start_time = time.time()
    knn.fit(X_train, y_train)
    knn.train_time_ = time.time() - start_time
    print(f"  Trained KNN with k={n_neighbors} in {knn.train_time_:.4f}s")
    return knn


def train_logistic_regression(X_train, y_train, feature_names=None,
                               C=1.0):
    """Train logistic regression and display feature coefficients."""
    lr = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    print(f"  Trained Logistic Regression (C={C:.4f})")

    if feature_names is not None:
        # Extract the actual model from the pipeline
        model = lr
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)

        print("\n  Feature Coefficients (sorted by magnitude):")
        for _, row in coef_df.iterrows():
            direction = '+' if row['Coefficient'] > 0 else '-'
            print(f"    {direction} {row['Feature']:30s} {row['Coefficient']:+.4f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in coef_df['Coefficient']]
        ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, alpha=0.8)
        ax.set_xlabel('Coefficient Value')
        ax.set_title('Logistic Regression Feature Coefficients')
        ax.axvline(0, color='black', linewidth=0.5)
        ax.invert_yaxis()
        plt.tight_layout()
        _save_fig('lr_coefficients')
        plt.show()

    return lr


def print_classification_report(model, X_test, y_test, model_name: str):
    """Print classification report for a model."""
    y_pred = model.predict(X_test)
    print(f"\n  Classification Report — {model_name}")
    print(f"  {'-'*50}")
    if hasattr(model, 'train_time_'):
        print(f"  Train Time: {model.train_time_:.4f} seconds")
    print(classification_report(y_test, y_pred, zero_division=0))


def plot_roc_curves(models: dict, X_test, y_test):
    """Plot ROC curves for multiple models on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 8))

    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    _save_fig('roc_curves')
    plt.show()


def plot_confusion_matrices(models: dict, X_test, y_test):
    """Plot confusion matrices for multiple models side by side."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Suspicious'],
                    yticklabels=['Normal', 'Suspicious'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name}')

    plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig('confusion_matrices')
    plt.show()
