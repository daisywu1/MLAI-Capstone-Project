# Anomalous Authentication Behavior Detection Using Machine Learning

**AI/ML Capstone Project — Daisy Wu**

> **Documentation:**  
> - [README.md](README.md) — Project overview and quick start (this file)  
> - [FINAL_ANALYSIS_REPORT.md](FINAL_ANALYSIS_REPORT.md) — Detailed final analysis report  
> - [project_problem_statement.md](project_problem_statement.md) — Original problem statement  

## About the Author
Hi, I'm Daisy Wu. I am a software engineer with over 20 years of experience in application development. I currently work for a cybersecurity company, which inspired me to choose this specific domain for my AI/ML capstone project. You can connect with me or view my professional background on [LinkedIn](https://www.linkedin.com/in/hire-daisy-wu/).

---

## 1. Problem Statement & Overview
This project aims to answer a core cybersecurity question: **Can machine learning techniques effectively identify and classify anomalous authentication behavior in enterprise security logs?** 

By analyzing historical access data, the goal is to build a predictive model that distinguishes normal user login patterns from potentially suspicious activity (such as credential stuffing, brute-force attempts, or compromised account usage) to enable faster detection and response.

I utilized the **Los Alamos National Laboratory (LANL) Cyber Security Dataset** (available at https://csr.lanl.gov/data/cyber1/) to see if ML could effectively isolate these rare events and provide actionable insights. The pipeline covers the full ML lifecycle: data preprocessing, exploratory analysis, feature engineering, unsupervised learning, time series analysis, supervised classification, and interpretable rule extraction.

## 2. Dataset & Data Engineering
The project uses authentication log data from the LANL dataset, representing 58 consecutive days of de-identified events from their internal network.

| Statistic | Value |
|-----------|-------|
| Total authentication events | ~1.6 billion |
| Duration | 58 days |
| Unique users | 12,425 |
| Unique computers | 17,684 |
| Known red team (malicious) events | 749 |
| Class imbalance ratio | ~0.00005% positive |

My data pipeline processed a stratified sample of the dataset, yielding ~638,500 log events, of which only 702 were labeled as malicious (an extreme class imbalance of roughly 0.1%). To address this before model training, I applied the **SMOTE** (Synthetic Minority Over-sampling Technique) strategy to synthesize new minority class examples, preventing models from simply defaulting to the majority class.

From an engineering perspective, feeding raw log strings (like `user@domain`) directly into a model is ineffective. The real work was in feature engineering—transforming raw logs into behavioral metrics. I constructed 16 behavioral features, with the most discriminative being:
1. `unique_dst_computers_24h`: The count of distinct destination computers a user accessed in a rolling 24-hour window. (This proved to be the most critical indicator of lateral movement).
2. `hour`: The time of day the event occurred.
3. `login_count_24h`: The total volume of logins for a user over 24 hours.

## 3. Techniques Applied
The analysis pipeline implements the following techniques to address the problem statement:

| Technique | Application |
|-----------|-------------|
| **Clustering and PCA** | Used PCA to reduce the 16-dimensional feature space to 2 components for visualization. Applied K-Means clustering to discover 3 natural behavioral groupings. |
| **Feature Engineering** | Engineered rolling-window behavioral features (counts, ratios, velocities) from raw logs to capture historical patterns without overfitting. |
| **Data Balancing (SMOTE)** | Applied Synthetic Minority Over-sampling Technique (SMOTE) to the training set to synthesize rare red team events and address the extreme class imbalance before model training. |
| **Time Series Analysis** | Decomposed hourly event volume to model temporal trends (seasonality) and detected sudden deviations via rolling z-scores. |
| **Model Selection & Hyperparameter Tuning** | Compared models using `GridSearchCV` with 5-fold cross-validation. Tuned L1/L2 regularization for Logistic Regression, max depth for Decision Trees, and estimators for Random Forest to handle high-dimensional spaces and prevent overfitting. |
| **Baseline Comparison** | Established a performance floor using a DummyClassifier (stratified random guessing) to ensure all experimental models provide real predictive value. |
| **Dual-Metric Evaluation** | Evaluated models using both **ROC AUC** (threshold-independent ranking) and **F2-Score** (threshold-dependent, Recall weighted 2x) to capture complementary views of model performance. |
| **Classification (KNN, LR, RF, SVM, DT)** | Built and evaluated baseline and advanced classifiers across ROC AUC, F2-Score, Recall, Precision, and Training Time. |
| **Model Interpretability (SHAP & Rules)** | Extracted human-readable rule sets from Decision Trees and used SHAP (SHapley Additive exPlanations) summary plots to explain the feature impact on model predictions. |

## 4. Model Evaluation & Conclusion

### Evaluation Strategy: ROC AUC + F2-Score
In cybersecurity threat detection, no single metric tells the full story. I evaluate models using two complementary metrics:

| Metric | What it measures | Strengths | Limitation |
|--------|-----------------|-----------|------------|
| **ROC AUC** | Overall ability to rank threats above normal events across *all* thresholds | Threshold-independent; unaffected by class imbalance | Does not reflect real-world operating performance at a specific decision threshold |
| **F2-Score** | Detection quality at the default threshold, weighting **Recall 2x** over Precision (β=2) | Directly reflects production behavior; penalizes missed threats more than false alarms | Threshold-dependent; a single number for one operating point |

Using both metrics reveals important insights. A model can have a high ROC AUC (good ranking ability) but a poor F2-Score (poor detection at the default threshold), which is exactly what happened with KNN. For production deployment decisions, **F2-Score is the primary guide** because it reflects actual detection performance.

I evaluated seven classification approaches, including a **Baseline (Stratified DummyClassifier)** as the performance floor.

| Model | ROC AUC | F2-Score | Recall | Precision | Train Time | Notes |
|-------|---------|----------|--------|-----------|------------|-------|
| **Baseline (Stratified)** | ~0.50 | ~0.01 | ~0.50 | ~0.001 | <0.01s | Random guessing (performance floor) |
| **K-Nearest Neighbors** | **0.9982** | ~0.18 | ~0.14 | ~0.50 | ~0.5s | Best AUC, but worst F2 — defaults to majority class |
| **Random Forest** | 0.9818 | ~0.40 | ~0.80 | ~0.12 | ~2.0s | Robust ensemble |
| **Logistic Regression (L2)** | 0.9660 | ~0.35 | ~0.88 | ~0.08 | ~0.2s | Highest Recall |
| **Support Vector Machine** | 0.9651 | ~0.34 | ~0.87 | ~0.08 | ~1.5s | Similar to LR, slower |
| **Decision Tree** | 0.9600 | **~0.38** | ~0.84 | ~0.10 | **<0.1s** | Best F2 + fastest + interpretable |

**Key observation**: KNN has the best ROC AUC (0.9982) but the worst F2-Score (~0.18). This demonstrates that a model can be excellent at *ranking* events but terrible at *detecting* threats at the operating threshold. This is why both metrics are essential.

### Production Recommendation: The Best Model
After analyzing the trade-off between ROC AUC, F2-Score, training efficiency, and interpretability, I recommend **Decision Tree** as the optimal model for production deployment.

**Justification:**
1. **Best F2-Score (~0.38)**: Achieves the best balance of catching threats while maintaining reasonable precision at the default threshold.
2. **Strong ROC AUC (0.96)**: Confirms the model has excellent overall discrimination ability.
3. **Fastest Training (<0.1s)**: Critical for adapting to evolving attack patterns via rapid retraining.
4. **Interpretability**: Produces explicit boolean rules (e.g., `if unique_dst_computers_24h > X and hour < Y`) that can be directly implemented as SIEM alerts.

**Alternative: Logistic Regression (L2)** — Offers the highest Recall (~88%) when minimizing missed threats is the absolute priority and the security team can handle more false alarms.

**Why NOT KNN?** Despite the highest ROC AUC (0.9982), KNN has the worst F2-Score (~0.18) because it defaults to predicting "normal." AUC alone is misleading for imbalanced classification.

### Key Takeaways
1. **Use complementary metrics**: ROC AUC measures ranking quality; F2-Score measures detection quality. Both are needed.
2. **Feature Engineering is King**: The success of this project was driven by engineering stateful, rolling-window behavioral features that captured user patterns.
3. **Beware of misleading metrics**: KNN's high AUC was deceptive — it failed on F2-Score. Always evaluate on the metric that reflects your production objective.
4. **Interpretability matters**: For production security systems, a Decision Tree's exportable rules are more valuable than a marginally better but opaque model.

---

## Project Structure & Quick Start

### Compute Environment & Performance
This project is highly compute-intensive, requiring full-file scans of a 7.2 GB compressed dataset (~1.6 billion rows) and complex rolling-window feature engineering. 

**Local Development Environment:**
* **Machine:** Apple Mac (M4 Pro Chip)
* **CPU:** 14 Cores (10 Performance, 4 Efficiency)
* **Memory:** 48 GB RAM
* **OS:** macOS (Darwin 25.3.0)

**Execution Time:**
Running the entire Jupyter notebook pipeline end-to-end (`Kernel → Restart & Run All`) takes approximately **25–30 minutes** on the hardware described above. The most time-consuming steps are the full-file vectorized scan to capture all rare red team events (~15-20 minutes) and the rolling-window behavioral feature engineering.

```
MLAI-Capstone-Project/
├── README.md                              # This file
├── FINAL_ANALYSIS_REPORT.md               # Detailed final analysis report
├── project_problem_statement.md           # Original problem statement
├── requirements.txt                       # Python dependencies
├── notebooks/
│   └── capstone_analysis.ipynb            # Main analysis notebook (run this)
├── src/                                   # Reusable Python modules
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
│   └── processed/                         # Generated parquet files
├── images/                                # Saved plot images
└── models/                                # Saved trained models
```

### Running the Project

1. **Clone and setup environment:**
```bash
cd MLAI-Capstone-Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Download Data:**
Download `auth.txt.gz` and `redteam.txt.gz` from https://csr.lanl.gov/data/cyber1/ and place them in the `data/raw/` directory.

3. **Launch the notebook:**
```bash
cd notebooks
jupyter notebook capstone_analysis.ipynb
# Select kernel: "Python 3 (Capstone)"
# Run: Kernel → Restart & Run All
```

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