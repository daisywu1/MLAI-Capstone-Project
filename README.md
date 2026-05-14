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
| **Model Selection & Hyperparameter Tuning** | Compared models using `GridSearchCV` with 5-fold cross-validation. **Logistic Regression/SVM** tuned `C` to balance regularization against overfitting; **Decision Trees** tuned `max_depth` to balance interpretability vs complex patterns; **Random Forest** tuned `n_estimators` & `max_depth` to balance robustness against elapsed training time. |
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

I evaluated seven classification approaches, including a **Baseline (Stratified DummyClassifier)** as the performance floor. The metrics below represent the **true test set performance** to ensure no data leakage from the SMOTE balancing process.

| Model | ROC AUC | F2-Score | Recall | Precision | Train Time | Notes |
|-------|---------|----------|--------|-----------|------------|-------|
| **Baseline (Stratified)** | ~0.50 | ~0.00 | ~0.50 | <0.01 | <0.01s | Random guessing (performance floor) |
| **K-Nearest Neighbors** | 0.8462 | **0.2771** | 0.6500 | **0.0841** | **~0.02s** | Best out-of-box F2 & precision |
| **Random Forest** | **0.9786** | 0.1849 | 0.8357 | 0.0449 | ~12.2s | Most robust (Best AUC, high Recall) |
| **Logistic Regression** | 0.9522 | 0.0434 | **0.8786** | 0.0090 | ~0.8s | Highest Recall, but too many false alarms |
| **Decision Tree** | 0.9275 | 0.1478 | 0.7214 | 0.0354 | ~3.7s | Good interpretable baseline |
| **Support Vector Machine** | N/A | 0.0429 | 0.8786 | 0.0089 | ~1.1s | Similar to LR, slower |

**Key observation**: Linear models (LR, SVM) achieved the highest Recall (~88%) but failed catastrophically on Precision (<1%), generating an unacceptable volume of false alarms. **KNN** achieved the best F2-Score out of the box by preserving Precision (8.4%), while **Random Forest** proved to be the most robust overall with the highest ROC AUC (97.8%) and a strong Recall (83.5%).

### Production Recommendation: The Best Model
After analyzing the trade-off between ROC AUC, F2-Score, training efficiency, and interpretability, I recommend **Random Forest** as the optimal model for production deployment, provided its operating threshold is tuned.

**Justification:**
1. **Best ROC AUC (0.978)**: Confirms the model has excellent overall discrimination ability between normal and suspicious events.
2. **High Recall (83.5%)**: Safely catches the vast majority of threats.
3. **Threshold Tunability**: While KNN has a better F2-Score *at the default threshold*, Random Forest's superior underlying ranking capability (AUC) means we can tune the decision threshold to easily match or exceed KNN's operational performance.

**Alternative: KNN** — Offers the best out-of-the-box F2-Score (0.277) and Precision (8.4%) without tuning, minimizing false alarms while still catching 65% of threats.

### Key Takeaways
1. **SMOTE and Data Leakage**: It is critical to evaluate models on a pristine test set. Cross-validation on SMOTE-augmented data yields wildly optimistic scores; true test performance reveals the real operational capability.
2. **Feature Engineering is King**: Behavioral features that capture rolling-window counts and velocities were the primary drivers of model performance.
3. **Use Complementary Metrics**: High Recall is useless if Precision falls below 1% (generating 100+ false alarms per real threat). Evaluating both F2-Score and ROC AUC ensures models are both operationally viable and robust.
4. **Training Elapsed Time**: The time it takes to train a model heavily impacts operations. While Random Forest takes ~12 seconds, simpler models like KNN (~0.02s) and Decision Trees (~3.7s) allow for rapid retraining on new data, providing faster response to emerging threats.

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