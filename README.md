# Anomalous Authentication Behavior Detection Using Machine Learning

**AI/ML Capstone Project — Daisy Wu**

This project applies machine learning techniques to identify and classify anomalous authentication behavior in enterprise security logs using the [Los Alamos National Laboratory (LANL) Cyber Security Dataset](https://csr.lanl.gov/data/cyber1/). The pipeline covers the full ML lifecycle: data preprocessing, exploratory analysis, feature engineering, unsupervised learning, time series analysis, supervised classification, and interpretable rule extraction.

## Project Structure

```
MLAI-Capstone-Project/
├── README.md                              # This file
├── DAISY_WU_AI_ML_Capstone_Project.md     # Problem statement
├── project_conclusion.md                  # Final conclusions and findings
├── requirements.txt                       # Python dependencies
├── .gitignore
├── notebooks/
│   └── capstone_analysis.ipynb            # Main analysis notebook (run this)
├── src/                                   # Reusable Python modules
│   ├── __init__.py
│   ├── data_preprocessing.py              # Data loading, cleaning, labeling
│   ├── feature_engineering.py             # Behavioral feature construction
│   ├── eda.py                             # Exploratory data analysis plots
│   ├── clustering_pca.py                  # PCA & K-Means clustering
│   ├── time_series.py                     # Seasonal decomposition & anomaly detection
│   ├── model_selection.py                 # Cross-validation, regularization, evaluation
│   ├── classification.py                  # KNN & Logistic Regression classifiers
│   ├── decision_trees.py                  # Decision tree training, rules, visualization
│   └── utils.py                           # Shared utilities & constants
├── data/
│   ├── raw/                               # LANL dataset files (not in git)
│   │   ├── auth.txt.gz                    # Authentication events (~7.2 GB)
│   │   └── redteam.txt.gz                 # Red team labels (~4.8 KB)
│   └── processed/                         # Generated parquet files
├── images/                                # Saved plot images
└── models/                                # Saved trained models
```

## Dataset

**LANL Comprehensive, Multi-Source Cyber-Security Events** — 58 consecutive days of de-identified authentication events from Los Alamos National Laboratory's internal network.

| Statistic | Value |
|-----------|-------|
| Total authentication events | ~1.6 billion |
| Duration | 58 days |
| Unique users | 12,425 |
| Unique computers | 17,684 |
| Known red team (malicious) events | 749 |
| Class imbalance ratio | ~0.00005% positive |

### Required Files

Download from [https://csr.lanl.gov/data/cyber1/](https://csr.lanl.gov/data/cyber1/):

| File | Description | Size |
|------|-------------|------|
| `auth.txt.gz` | Authentication events | ~7.2 GB |
| `redteam.txt.gz` | Known red team compromise events | ~4.8 KB |

Place both files in `data/raw/` before running the notebook.

### Data Format

**auth.txt** (comma-delimited):
```
time, src_user@domain, dst_user@domain, src_computer, dst_computer, auth_type, logon_type, auth_orientation, success/failure
```

**redteam.txt** (comma-delimited):
```
time, user@domain, source_computer, destination_computer
```

## Quick Start

```bash
# 1. Clone and enter the project
cd MLAI-Capstone-Project

# 2. Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Place LANL data files in data/raw/
#    (auth.txt.gz and redteam.txt.gz)

# 4. Launch the notebook
cd notebooks
jupyter notebook capstone_analysis.ipynb
# Select kernel: "Python 3 (Capstone)"
# Run: Kernel → Restart & Run All
```

## Analysis Pipeline

The notebook implements all seven techniques from the capstone problem statement:

| Section | Course Module | Technique | Purpose |
|---------|--------------|-----------|---------|
| 1–2. Data Loading | — | Stratified sampling, cleaning | Load ~1M rows spread across 58 days |
| 3. EDA | — | Statistical summaries, 7 visualizations | Understand data distributions and patterns |
| 4. Feature Engineering | Module 8 | Rolling windows, ratios, velocity | Build 16 behavioral features per event |
| 5. PCA & Clustering | Module 6 | PCA, K-Means, silhouette analysis | Discover natural behavioral groupings |
| 6. Time Series | Module 10 | Seasonal decomposition, z-score | Detect temporal anomalies |
| 7. Regularization | Modules 9 & 15 | Ridge/Lasso, CV, learning curves | Tune models, prevent overfitting |
| 8. Classification | Modules 11 & 13 | KNN, Logistic Regression | Baseline classifiers with ROC/CM |
| 9. Decision Trees | Module 14 | Tree visualization, rule extraction | Interpretable rules for analysts |
| 10. Comparison | — | Side-by-side evaluation | Select best model |

### Engineered Features (16 total)

| Category | Features |
|----------|----------|
| Temporal | `hour`, `day_of_week`, `is_off_hours`, `is_weekend` |
| Outcome | `success` |
| Frequency | `login_count_1h`, `login_count_6h`, `login_count_24h` |
| Failure | `fail_count_1h`, `fail_ratio_24h` |
| Diversity | `unique_dst_computers_24h`, `unique_auth_types_24h`, `unique_logon_types_24h` |
| Behavioral | `time_since_last_login`, `is_new_dst_computer`, `events_per_minute_10m` |

## Output Artifacts

After a successful run:

| Output | Location | Description |
|--------|----------|-------------|
| Cleaned data | `data/processed/auth_cleaned.parquet` | Parsed and labeled events |
| Feature matrix | `data/processed/auth_features.parquet` | 16 engineered features + target |
| Visualizations | `images/*.png` | ~15–20 plots (EDA, PCA, ROC, tree, etc.) |
| Trained models | `models/*.joblib` | Serialized best models |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: auth.txt.gz` | Place it in `data/raw/` |
| `ModuleNotFoundError` | Activate venv: `source venv/bin/activate` |
| Wrong kernel | Switch to **Python 3 (Capstone)** |
| Memory error | Reduce `SAMPLE_SIZE` in cell 4 (e.g., 500,000) |
| Slow feature engineering | Reduce `SAMPLE_SIZE`; 500K rows takes ~2 min |

## Citation

```
A. D. Kent, "Comprehensive, Multi-Source Cybersecurity Events,"
Los Alamos National Laboratory, 2015.
http://dx.doi.org/10.17021/1179829

A. D. Kent, "Cybersecurity Data Sources for Dynamic Network Research,"
in Dynamic Networks in Cybersecurity, Imperial College Press, 2015.
```

## License

This project is for academic purposes as part of an AI/ML capstone course. The LANL dataset is published under a CC0 public domain dedication by Los Alamos National Laboratory.
