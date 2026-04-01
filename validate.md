# How to Test and Validate the Notebook

## Project Architecture

The `src/` files are **helper modules** that the notebook imports and calls. You do not run them directly.

```
notebooks/capstone_analysis.ipynb   <-- YOU RUN THIS
    ├── imports src/data_preprocessing.py   (loads auth.txt.gz, redteam.txt.gz)
    ├── imports src/feature_engineering.py   (builds behavioral features)
    ├── imports src/eda.py                   (all EDA plots)
    ├── imports src/clustering_pca.py        (PCA + K-Means)
    ├── imports src/time_series.py           (decomposition + anomaly detection)
    ├── imports src/model_selection.py       (cross-validation, regularization)
    ├── imports src/classification.py        (KNN + Logistic Regression)
    ├── imports src/decision_trees.py        (tree training, rules, visualization)
    └── imports src/utils.py                 (shared utilities, train/test split)
```

When you run a cell like `df_raw = load_auth_data(nrows=SAMPLE_SIZE)`, it calls the function defined in `src/data_preprocessing.py`. The notebook's first code cell does `sys.path.insert(0, '..')` and imports everything from `src/`.

## What to do with the `src/` files

**Nothing manually.** They exist so the notebook stays clean and readable. You only interact with the notebook.

## Step 1: Run the Notebook Top-to-Bottom

1. Make sure `auth.txt.gz` and `redteam.txt.gz` are both in `data/raw/`
2. Open `notebooks/capstone_analysis.ipynb` in Jupyter
3. Select kernel: **Python 3 (Capstone)**
4. **Kernel → Restart & Run All**
5. Wait ~15–20 minutes for all cells to complete

If every cell executes without errors, the project works.

## Step 2: Spot-Check the Outputs

After a successful run, verify:

### Files created

| What | Where | Expected |
|------|-------|----------|
| Cleaned data | `data/processed/auth_cleaned.parquet` | File exists, non-zero size |
| Feature matrix | `data/processed/auth_features.parquet` | File exists, non-zero size |
| All plots | `images/` | ~15–20 PNG files |
| Trained models | `models/` | 2 `.joblib` files |

### Quick terminal check

```bash
ls -lh data/processed/
ls -lh images/
ls -lh models/
```

### In the notebook itself

- **No tracebacks** — every cell should have output (tables, stats, or plots)
- **Section 3 (EDA)** — 7 inline plots visible
- **Section 5 (PCA)** — scatter plots with cluster colors and red team overlay
- **Section 6 (Time Series)** — decomposition plot + anomaly markers
- **Section 8 (Classification)** — ROC curves and confusion matrices
- **Section 9 (Decision Trees)** — tree visualization + printed rules
- **Section 10 (Comparison)** — all models evaluated with metrics printed
- **Section 11 (Conclusions)** — fill in findings after reviewing results

## If Something Fails

- **"File not found: auth.txt.gz"** → Copy it to `data/raw/`
- **ModuleNotFoundError** → Activate venv: `source venv/bin/activate`, then relaunch Jupyter
- **Wrong kernel** → Switch to **Python 3 (Capstone)** in Kernel menu
- **Memory error on load** → Reduce `SAMPLE_SIZE` in cell 4 (e.g., from 1,000,000 to 500,000)
- **Feature engineering too slow** → Reduce `SAMPLE_SIZE`; 500K rows takes ~2 min
