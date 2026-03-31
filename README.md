# Anomalous Authentication Behavior Detection Using Machine Learning

**AI/ML Capstone Project — Daisy Wu**

This project applies machine learning techniques to identify and classify anomalous authentication behavior in enterprise security logs using the [Los Alamos National Laboratory (LANL) Cyber Security Dataset](https://csr.lanl.gov/data/cyber1/).

## Project Structure

```
MLAI-Capstone-Project/
├── README.md
├── requirements.txt
├── DAISY_WU_AI_ML_Capstone_Project.md   # Problem statement
├── notebooks/
│   └── capstone_analysis.ipynb          # Main analysis notebook
├── src/                                 # Python source modules
│   ├── __init__.py
│   ├── data_preprocessing.py            # Data loading & cleaning
│   ├── feature_engineering.py           # Behavioral feature construction
│   ├── eda.py                           # Exploratory data analysis plots
│   ├── clustering_pca.py               # PCA & K-Means clustering
│   ├── time_series.py                  # Time series decomposition & anomaly detection
│   ├── model_selection.py              # Cross-validation, regularization, evaluation
│   ├── classification.py              # KNN & Logistic Regression classifiers
│   ├── decision_trees.py             # Decision tree training, visualization, rules
│   └── utils.py                       # Shared utilities & constants
├── data/
│   ├── raw/                            # Place LANL dataset files here
│   │   ├── (auth.txt.gz)              # Authentication events (~9 GB compressed)
│   │   └── (redteam.txt.gz)           # Red team labels
│   └── processed/                      # Generated parquet files (auto-created)
├── images/                             # Saved plot images (auto-created)
└── models/                             # Saved trained models (auto-created)
```

## Dataset

This project uses the **LANL Comprehensive, Multi-Source Cyber-Security Events** dataset:

> A. D. Kent, "Comprehensive, Multi-Source Cybersecurity Events,"
> Los Alamos National Laboratory, 2015.
> http://dx.doi.org/10.17021/1179829

### Required Files

Download from [https://csr.lanl.gov/data/cyber1/](https://csr.lanl.gov/data/cyber1/):

| File | Description | Size |
|------|-------------|------|
| `auth.txt.gz` | Authentication events (~1.6B rows) | ~9 GB |
| `redteam.txt.gz` | Known red team (malicious) events | ~1 KB |

Place both files in `data/raw/` before running the notebook.

### auth.txt Format

```
time, src_user@domain, dst_user@domain, src_computer, dst_computer, auth_type, logon_type, auth_orientation, success/failure
```

Example:
```
1,C625$@DOM1,U147@DOM1,C625,C625,Negotiate,Batch,LogOn,Success
```

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the LANL dataset

Place `auth.txt.gz` and `redteam.txt.gz` in `data/raw/`.

### 4. Run the notebook

```bash
cd notebooks
jupyter notebook capstone_analysis.ipynb
```

## Analysis Pipeline

| Section | Module | Technique |
|---------|--------|-----------|
| 1. Data Loading | — | Load & preprocess LANL auth.txt |
| 2. EDA | — | Summary stats, temporal patterns, distributions |
| 3. Feature Engineering | Module 8 | Rolling windows, failure ratios, velocity metrics |
| 4. PCA & Clustering | Module 6 | Dimensionality reduction, K-Means |
| 5. Time Series | Module 10 | Seasonal decomposition, z-score anomalies |
| 6. Regularization | Module 9 | Ridge/Lasso tuning, model comparison |
| 7. Classification | Modules 11 & 13 | KNN, Logistic Regression |
| 8. Decision Trees | Module 14 | Interpretable rules for analysts |
| 9. Gradient Descent | Module 15 | Solver/convergence analysis |
| 10. Comparison | — | Side-by-side model evaluation |

## Output

- **images/**: All generated plots (EDA charts, PCA scatter, ROC curves, decision tree, etc.)
- **models/**: Serialized trained models (`.joblib`)
- **data/processed/**: Intermediate parquet files for fast reloading
