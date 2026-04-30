# Anomalous Authentication Behavior Detection Using Machine Learning

**AI/ML Capstone Project — Daisy Wu**

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
| **Classification (KNN, LR, RF, SVM, DT)** | Built and evaluated baseline and advanced classifiers, comparing ROC AUC, Accuracy, F1-Score, and Training Time. |
| **Model Interpretability (SHAP & Rules)** | Extracted human-readable rule sets from Decision Trees and used SHAP (SHapley Additive exPlanations) summary plots to explain the feature impact on model predictions. |

## 4. Model Evaluation & Conclusion

### Why RECALL is the Primary Metric
In cybersecurity threat detection, **Recall** (also known as Sensitivity or True Positive Rate) is the most critical metric. A missed threat (False Negative) — where a compromised account goes undetected — can result in catastrophic data breaches, financial losses, and regulatory penalties. In contrast, a false alarm (False Positive) only requires additional analyst review. Therefore, **optimizing for Recall is paramount**; we would rather investigate 100 false alarms than miss a single real intrusion.

I evaluated six different classification approaches, including a **Baseline (Stratified DummyClassifier)** to establish a performance floor. All experimental models must beat this baseline to be considered useful.

| Model | RECALL | ROC AUC | F1-Score | Train Time | Notes |
|-------|--------|---------|----------|------------|-------|
| **Baseline (Stratified)** | ~0.50 | ~0.50 | ~0.01 | <0.01s | Random guessing (performance floor) |
| **Logistic Regression (L2)** | ~0.88 | 0.9660 | ~0.15 | ~0.2s | High recall, strong linear separator |
| **Support Vector Machine** | ~0.87 | 0.9651 | ~0.14 | ~1.5s | Similar to LR, slower training |
| **Decision Tree** | ~0.84 | 0.9600 | ~0.18 | <0.1s | Fast, interpretable rules |
| **Random Forest** | ~0.80 | 0.9818 | ~0.20 | ~2.0s | Robust ensemble, handles non-linearity |
| **K-Nearest Neighbors** | ~0.14 | 0.9982 | 0.25 | ~0.5s | High AUC but very low recall — defaults to majority class |

All experimental models vastly outperform the baseline, confirming that the engineered features contain real predictive signal for detecting anomalous authentication behavior. **Notably, KNN achieves the highest AUC but fails catastrophically on Recall**, demonstrating why AUC alone is insufficient for security-critical applications. 

### Production Recommendation: The Best Model
After analyzing the trade-off between **Recall** (primary metric), training efficiency, and interpretability, I selected **Logistic Regression (L2)** as the optimal model for production deployment, with **Decision Tree** as a close second for scenarios requiring interpretable rules.

**Justification:**
1. **Highest Recall (~88%)**: Logistic Regression catches the highest percentage of actual threats, minimizing dangerous False Negatives. In security, missing a threat is far more costly than a false alarm.
2. **Strong ROC AUC (0.9660)**: The model maintains excellent overall discrimination ability between normal and anomalous behavior.
3. **Fast Training (~0.2s)**: Quick retraining enables the model to adapt to evolving attack patterns in near real-time.
4. **Balanced Class Weights**: The `class_weight='balanced'` parameter ensures the model prioritizes the rare positive class.

**Alternative: Decision Tree** — While slightly lower Recall (~84%), the Decision Tree offers a unique advantage: **interpretability**. Its explicit boolean rules (e.g., `if unique_dst_computers_24h > X and hour < Y`) can be directly exported and implemented as deterministic alerts in an existing SIEM system, bridging the gap between ML experimentation and deployable security logic.

**Why NOT KNN?** Despite achieving the highest ROC AUC (0.9982), KNN has a catastrophically low Recall (~14%). It essentially defaults to predicting the majority class ("normal"), making it useless for detecting actual threats.

### Key Takeaways
1. **Recall > Accuracy**: In security classification, optimizing for Recall is critical. A model with 99.9% accuracy that misses all threats is worthless.
2. **Feature Engineering is King**: The success of this project wasn't driven by choosing a complex model, but by engineering stateful, rolling-window behavioral features that captured user patterns.
3. **Beware of Misleading Metrics**: KNN's high AUC was deceptive—it failed on the metric that matters most (Recall). Always evaluate models on business-critical metrics.
4. **Interpretability Matters**: For production security systems, a slightly less accurate but interpretable model (Decision Tree) may be more valuable than a black-box model.

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
├── DAISY_WU_AI_ML_Capstone_Project.md     # Problem statement
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