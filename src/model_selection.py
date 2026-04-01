"""Model selection, regularization tuning, and evaluation utilities."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from src.utils import IMAGES_DIR, MODELS_DIR


def _save_fig(name: str):
    path = os.path.join(IMAGES_DIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                   cv: int = 5) -> pd.DataFrame:
    """Cross-validate multiple candidate models and compare.

    Returns a DataFrame with mean and std scores for each model.
    """
    models = {
        'Logistic Regression (L2)': LogisticRegression(
            max_iter=1000, solver='lbfgs', random_state=42, class_weight='balanced'),
        'Logistic Regression (L1)': LogisticRegression(
            penalty='l1', max_iter=1000, solver='saga', random_state=42, class_weight='balanced'),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree (depth=5)': DecisionTreeClassifier(
            max_depth=5, random_state=42, class_weight='balanced'),
    }

    n_positive = int(y_train.sum())
    if n_positive < cv:
        print(f"  Warning: Only {n_positive} positive samples. Reducing CV folds to {max(2, n_positive)}.")
        cv = max(2, n_positive)
        
    if cv < 2:
        print("  Warning: Not enough positive samples for cross-validation. Skipping model comparison.")
        return pd.DataFrame()

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = []
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv_strategy,
                                     scoring='roc_auc', n_jobs=-1)
            results.append({
                'Model': name,
                'Mean ROC AUC': scores.mean(),
                'Std ROC AUC': scores.std(),
            })
            print(f"  {name}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
        except Exception as e:
            print(f"  {name}: Failed CV ({str(e)})")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results).sort_values('Mean ROC AUC', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df['Model'], df['Mean ROC AUC'], xerr=df['Std ROC AUC'],
            color='steelblue', alpha=0.8, capsize=5)
    ax.set_xlabel('ROC AUC (cross-validated)')
    ax.set_title('Model Comparison')
    ax.set_xlim(0, 1)
    plt.tight_layout()
    _save_fig('model_comparison_cv')
    plt.show()

    return df


def tune_regularization(X_train: np.ndarray, y_train: np.ndarray,
                        cv: int = 5):
    """Tune the regularization strength (C) for logistic regression.

    Returns (best_model, results_df).
    """
    C_values = np.logspace(-4, 4, 20)
    results = []

    n_positive = int(y_train.sum())
    if n_positive < cv:
        cv = max(2, n_positive)
        
    if cv >= 2:
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        for C in C_values:
            lr = LogisticRegression(C=C, max_iter=1000, solver='lbfgs',
                                    random_state=42, class_weight='balanced')
            try:
                scores = cross_val_score(lr, X_train, y_train, cv=cv_strategy,
                                         scoring='roc_auc', n_jobs=-1)
                results.append({'C': C, 'Mean AUC': scores.mean(), 'Std AUC': scores.std()})
            except Exception:
                pass

    if not results:
        print("  Warning: CV failed. Using default C=1.0")
        best_model = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', random_state=42, class_weight='balanced')
        best_model.fit(X_train, y_train)
        return best_model, pd.DataFrame([{'C': 1.0, 'Mean AUC': 0.5, 'Std AUC': 0.0}])

    df = pd.DataFrame(results)
    best_idx = df['Mean AUC'].idxmax()
    best_C = df.loc[best_idx, 'C']
    print(f"  Best C = {best_C:.6f} (AUC = {df.loc[best_idx, 'Mean AUC']:.4f})")

    best_model = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs',
                                    random_state=42, class_weight='balanced')
    best_model.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(df['C'], df['Mean AUC'], 'o-', color='steelblue', linewidth=2)
    ax.fill_between(df['C'],
                    df['Mean AUC'] - df['Std AUC'],
                    df['Mean AUC'] + df['Std AUC'],
                    alpha=0.2, color='steelblue')
    ax.axvline(best_C, color='red', linestyle='--', alpha=0.7,
               label=f'Best C = {best_C:.4f}')
    ax.set_xlabel('Regularization Strength (C)')
    ax.set_ylabel('ROC AUC')
    ax.set_title('Logistic Regression: Regularization Tuning')
    ax.legend()
    plt.tight_layout()
    _save_fig('regularization_tuning')
    plt.show()

    return best_model, df


def plot_learning_curve(model, X_train, y_train, title: str):
    """Plot learning curve (train vs. validation score vs. training size)."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-',
            color='steelblue', label='Training score')
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1),
                    alpha=0.1, color='steelblue')
    ax.plot(train_sizes, val_scores.mean(axis=1), 'o-',
            color='darkorange', label='Validation score')
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1),
                    alpha=0.1, color='darkorange')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('ROC AUC')
    ax.set_title(f'Learning Curve: {title}')
    ax.legend()
    plt.tight_layout()
    _save_fig('learning_curve')
    plt.show()


def evaluate_model(model, X_test, y_test):
    """Print full evaluation metrics for a model on the test set."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")
    if y_prob is not None:
        print(f"  ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")


def save_model(model, filename: str):
    """Save a trained model to the models/ directory."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"  Saved model: {path}")
