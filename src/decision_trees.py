"""Decision tree classification, visualization, and rule extraction."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.utils import IMAGES_DIR


def _save_fig(name: str):
    path = os.path.join(IMAGES_DIR, f'{name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")


def tune_tree_depth(X_train, y_train, max_depth_range=range(2, 16), cv=5):
    """Find optimal max_depth via cross-validation.

    Returns (best_depth, results_df).
    """
    results = []
    n_positive = int(y_train.sum())
    if n_positive < cv:
        cv = max(2, n_positive)
        
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) if cv >= 2 else None

    for depth in max_depth_range:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42, class_weight='balanced')
        if cv_strategy:
            try:
                scores = cross_val_score(dt, X_train, y_train, cv=cv_strategy,
                                         scoring='roc_auc', n_jobs=-1)
                results.append({
                    'max_depth': depth,
                    'Mean AUC': scores.mean(),
                    'Std AUC': scores.std()
                })
                print(f"  depth={depth}: AUC = {scores.mean():.4f} ± {scores.std():.4f}")
            except Exception:
                pass

    if not results:
        print("  Warning: CV failed. Using default max_depth=5")
        return 5, pd.DataFrame([{'max_depth': 5, 'Mean AUC': 0.5, 'Std AUC': 0.0}])

    df = pd.DataFrame(results)
    best_idx = df['Mean AUC'].idxmax()
    best_depth = int(df.loc[best_idx, 'max_depth'])
    print(f"\n  Best max_depth = {best_depth} "
          f"(AUC = {df.loc[best_idx, 'Mean AUC']:.4f})")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['max_depth'], df['Mean AUC'], 'o-', color='steelblue', linewidth=2)
    ax.fill_between(df['max_depth'],
                    df['Mean AUC'] - df['Std AUC'],
                    df['Mean AUC'] + df['Std AUC'],
                    alpha=0.2, color='steelblue')
    ax.axvline(best_depth, color='red', linestyle='--', alpha=0.7,
               label=f'Best depth = {best_depth}')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('ROC AUC (CV)')
    ax.set_title('Decision Tree: Depth Tuning')
    ax.legend()
    plt.tight_layout()
    _save_fig('dt_depth_tuning')
    plt.show()

    return best_depth, df


def train_decision_tree(X_train, y_train, max_depth=5):
    """Train a decision tree classifier."""
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight='balanced')
    dt.fit(X_train, y_train)
    print(f"  Trained Decision Tree (max_depth={max_depth})")
    return dt

def visualize_tree(dt, feature_names, max_depth_display=3):
    """Visualize the decision tree."""
    model = dt.named_steps['decisiontreeclassifier'] if hasattr(dt, 'named_steps') else dt
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(model, feature_names=feature_names,
              class_names=['Normal', 'Suspicious'],
              filled=True, rounded=True, ax=ax,
              max_depth=max_depth_display, fontsize=9,
              proportion=True)
    ax.set_title('Decision Tree Visualization', fontsize=16)
    plt.tight_layout()
    _save_fig('dt_visualization')
    plt.show()

def extract_rules(dt, feature_names, max_depth=4):
    """Extract human-readable rules from the decision tree."""
    model = dt.named_steps['decisiontreeclassifier'] if hasattr(dt, 'named_steps') else dt
    rules_text = export_text(model, feature_names=feature_names, max_depth=max_depth)
    print("  Decision Tree Rules:")
    print("  " + "-" * 50)
    for line in rules_text.split('\n'):
        print(f"  {line}")
    return rules_text

def get_feature_importance(dt, feature_names):
    """Get and plot feature importance from the decision tree."""
    model = dt.named_steps['decisiontreeclassifier'] if hasattr(dt, 'named_steps') else dt
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("  Feature Importance:")
    for _, row in importance_df.iterrows():
        if row['Importance'] > 0:
            bar = '█' * int(row['Importance'] * 50)
            print(f"    {row['Feature']:30s} {row['Importance']:.4f} {bar}")

    fig, ax = plt.subplots(figsize=(10, 6))
    nonzero = importance_df[importance_df['Importance'] > 0]
    ax.barh(nonzero['Feature'], nonzero['Importance'],
            color='steelblue', alpha=0.8)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Decision Tree Feature Importance')
    ax.invert_yaxis()
    plt.tight_layout()
    _save_fig('dt_feature_importance')
    plt.show()

    return importance_df


