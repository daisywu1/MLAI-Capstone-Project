"""Model selection, regularization tuning, and evaluation utilities."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import shap
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
    warnings.filterwarnings('ignore')
    models = {
        'Logistic Regression (L2)': LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42, class_weight='balanced'),
        'Logistic Regression (L1)': LogisticRegression(penalty='l1', max_iter=1000, solver='saga', random_state=42, class_weight='balanced'),
        'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree (depth=5)': DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced', n_jobs=-1),
        'Support Vector Machine': LinearSVC(random_state=42, class_weight='balanced', max_iter=2000)
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
    warnings.filterwarnings('ignore')
    C_values = np.logspace(-4, 4, 20)
    results = []

    n_positive = int(y_train.sum())
    if n_positive < cv:
        cv = max(2, n_positive)
        
    if cv >= 2:
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        for C in C_values:
            lr = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=42, class_weight='balanced')
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

    best_model = LogisticRegression(C=best_C, max_iter=1000, solver='lbfgs', random_state=42, class_weight='balanced')
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
    warnings.filterwarnings('ignore')
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
    warnings.filterwarnings('ignore')
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    if hasattr(model, 'train_time_'):
        print(f"  Train Time: {model.train_time_:.4f} seconds")
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

def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray, cv: int = 5):
    """Use GridSearchCV to find the best hyperparameters for multiple models."""
    warnings.filterwarnings('ignore')
    print("Performing GridSearchCV to find the best hyperparameters...")
    n_positive = int(y_train.sum())
    if n_positive < cv:
        cv = max(2, n_positive)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if cv >= 2 else None

    pipelines = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        'SVM': LinearSVC(random_state=42, class_weight='balanced', max_iter=2000)
    }

    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10]},
        'Decision Tree': {'max_depth': [3, 5, 7, 10]},
        'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
        'SVM': {'C': [0.01, 0.1, 1, 10]}
    }

    best_models = {}
    for name in pipelines:
        print(f"\n  Tuning {name}...")
        if cv_strategy:
            grid = GridSearchCV(pipelines[name], param_grids[name], cv=cv_strategy, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            best_models[name] = grid.best_estimator_
            best_models[name].train_time_ = grid.refit_time_
            print(f"    Best params: {grid.best_params_}")
            print(f"    Best CV AUC: {grid.best_score_:.4f}")
            print(f"    Refit Time: {grid.refit_time_:.4f}s")
            
            if name == 'Logistic Regression':
                model = grid.best_estimator_
                print(f"    Best C: {model.C}")
            elif name == 'Decision Tree':
                model = grid.best_estimator_
                print(f"    Best max_depth: {model.max_depth}")
            elif name == 'Random Forest':
                model = grid.best_estimator_
                print(f"    Best n_estimators: {model.n_estimators}, max_depth: {model.max_depth}")
            elif name == 'SVM':
                model = grid.best_estimator_
                print(f"    Best C: {model.C}")
        else:
            start_time = time.time()
            pipelines[name].fit(X_train, y_train)
            best_models[name] = pipelines[name]
            best_models[name].train_time_ = time.time() - start_time
            print(f"    Not enough positive samples for CV. Using default params.")
            print(f"    Train Time: {best_models[name].train_time_:.4f}s")
    return best_models


def generate_shap_summary(best_pipeline, X_train, feature_names):
    """Generate and save a SHAP summary plot for the best model."""
    warnings.filterwarnings('ignore')
    print("\nGenerating SHAP summary plot for the best model...")
    model = best_pipeline

    # Sample background data to speed up SHAP calculation
    if isinstance(X_train, pd.DataFrame):
        X_sample = X_train.sample(n=min(500, len(X_train)), random_state=42)
    else:
        np.random.seed(42)
        indices = np.random.choice(X_train.shape[0], min(500, X_train.shape[0]), replace=False)
        X_sample = X_train[indices]

    X_sample_scaled = X_sample

    if type(model).__name__ in ['RandomForestClassifier', 'DecisionTreeClassifier']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample_scaled)
        if isinstance(shap_values, list):
            shap_values = shap_values[1] # Get values for the positive class
    else:
        explainer = shap.LinearExplainer(model, X_sample_scaled)
        shap_values = explainer.shap_values(X_sample_scaled)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample_scaled, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot: Feature Impact on Model Output')
    plt.tight_layout()
    _save_fig('shap_summary')
    plt.show()
