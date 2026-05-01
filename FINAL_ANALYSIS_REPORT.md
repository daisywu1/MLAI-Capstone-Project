# AI/ML Capstone Project — Final Analysis Report

**Author:** Daisy Wu  
**Date:** May 2026  
**Course:** AI/ML Capstone  

---

## 1. Problem Statement

**Research Question:**  
Can machine learning techniques effectively identify and classify anomalous authentication behavior within massive enterprise security logs to accurately detect compromised accounts and lateral movement?

**Business Context:**  
In cybersecurity, the cost of a missed threat (False Negative) can be catastrophic — data breaches, financial losses, regulatory penalties, and reputational damage. Security Operations Centers (SOCs) face the challenge of analyzing billions of authentication events to find the rare malicious activity hidden within normal traffic. This project explores whether ML can automate this "needle in a haystack" detection problem.

---

## 2. Data Sources and Structure

### Primary Dataset
**Los Alamos National Laboratory (LANL) Cyber Security Dataset**  
URL: https://csr.lanl.gov/data/cyber1/

| Attribute | Value |
|-----------|-------|
| Total authentication events | ~1.6 billion |
| Duration | 58 consecutive days |
| Unique users | 12,425 |
| Unique computers | 17,684 |
| Known malicious (red team) events | 749 |
| Class imbalance ratio | ~0.00005% positive |

### Data Files Used
- `auth.txt.gz` — Authentication events (timestamp, user, source, destination, auth type, logon type, success/failure)
- `redteam.txt.gz` — Ground truth labels for known compromise events

### Sampling Strategy
Due to the massive scale (1.6B rows), I processed a stratified sample of ~638,500 events that:
- Spans the full 58-day time range (ensures temporal diversity)
- Includes all 749 red team events (ensures positive class representation)
- Maintains the natural class distribution for realistic evaluation

### Feature Engineering
Raw log fields were transformed into 16 behavioral features:

| Feature | Description |
|---------|-------------|
| `unique_dst_computers_24h` | Distinct destination computers accessed in 24h (lateral movement indicator) |
| `login_count_1h/6h/24h` | Rolling login volume |
| `fail_ratio_24h` | Proportion of failed logins |
| `events_per_minute_10m` | Login velocity (burst detection) |
| `time_since_last_login` | Temporal gap analysis |
| `is_new_dst_computer` | First-time destination flag |
| `is_off_hours` | After-hours activity flag |
| `is_weekend` | Weekend activity flag |

---

## 3. Techniques Applied

### Data Preprocessing
- Parsed composite fields (`user@domain` → user, domain)
- Converted categorical outcomes to binary (success=1, failure=0)
- Derived temporal features (hour, day of week)
- Labeled red team events using vectorized matching

### Exploratory Data Analysis
- Login volume distribution over time
- Success/failure ratio analysis
- Hourly login patterns (discovered strong daily seasonality)
- User activity distribution (identified power-law pattern)
- Red team event timeline visualization

### Unsupervised Learning
- **PCA**: Reduced 16 features to 2 components for visualization; 8 components capture ~95% variance
- **K-Means Clustering**: Identified 3 natural behavioral groups (automated accounts, normal users, low-frequency accounts)

### Time Series Analysis
- Aggregated events to hourly time series
- Seasonal decomposition (24-hour period)
- Rolling z-score anomaly detection for temporal outliers

### Data Balancing
- **SMOTE** (Synthetic Minority Over-sampling Technique): Synthesized minority class examples in training set to address extreme class imbalance

### Supervised Learning Models
| Model | Hyperparameter Tuning |
|-------|----------------------|
| Logistic Regression (L1/L2) | C parameter via GridSearchCV |
| K-Nearest Neighbors | k via elbow method |
| Decision Tree | max_depth via GridSearchCV |
| Random Forest | n_estimators, max_depth via GridSearchCV |
| Support Vector Machine | C via GridSearchCV |

### Model Evaluation Strategy
**Dual-metric approach:**
- **ROC AUC**: Threshold-independent ranking quality
- **F2-Score**: Threshold-dependent detection quality (Recall weighted 2x over Precision)

Both metrics are necessary because a model can have excellent ranking (high AUC) but poor detection at the operating threshold (low F2).

### Model Interpretability
- **Confusion Matrices**: Visualized TP, FP, TN, FN for each model
- **SHAP Values**: Explained feature contributions to predictions
- **Decision Tree Rules**: Extracted human-readable boolean logic for SIEM deployment

---

## 4. Results

### Model Comparison

| Model | ROC AUC | F2-Score | Recall | Precision | Train Time |
|-------|---------|----------|--------|-----------|------------|
| **Baseline (Random)** | 0.50 | 0.01 | 0.50 | 0.001 | <0.01s |
| K-Nearest Neighbors | **0.9982** | 0.18 | 0.14 | 0.50 | ~0.5s |
| Random Forest | 0.9818 | 0.40 | 0.80 | 0.12 | ~2.0s |
| Logistic Regression (L2) | 0.9660 | 0.35 | 0.88 | 0.08 | ~0.2s |
| Support Vector Machine | 0.9651 | 0.34 | 0.87 | 0.08 | ~1.5s |
| **Decision Tree** | 0.9600 | **0.38** | 0.84 | 0.10 | **<0.1s** |

### Key Observations

1. **All models vastly outperform the baseline**, confirming the engineered features contain real predictive signal.

2. **KNN has the best AUC (0.9982) but the worst F2-Score (0.18)**. This demonstrates that excellent ranking ability does not guarantee good detection performance. KNN defaults to predicting "normal" for most events.

3. **Decision Tree provides the best balance**: Strong F2-Score (0.38), fastest training (<0.1s), and interpretable rules.

4. **Logistic Regression has the highest Recall (88%)** — best choice when minimizing missed threats is the absolute priority.

### Most Important Features (from Decision Tree and SHAP)
1. `unique_dst_computers_24h` — strongest indicator of lateral movement
2. `login_count_1h` — burst activity detection
3. `events_per_minute_10m` — velocity anomalies
4. `time_since_last_login` — temporal pattern breaks
5. `is_off_hours` — after-hours activity

### Interpretable Rules for Security Analysts
The Decision Tree produces rules that can be directly implemented in a SIEM:

```
IF unique_dst_computers_24h > 5 
   AND is_new_dst_computer = 1 
   AND is_off_hours = 1 
THEN → Flag as Suspicious

IF events_per_minute_10m > 3 
   AND fail_ratio_24h > 0.3 
THEN → Flag as Suspicious
```

---

## 5. Production Recommendation

**Selected Model: Decision Tree (max_depth=7)**

| Criterion | Value | Why it matters |
|-----------|-------|----------------|
| F2-Score | 0.38 | Best balance of detection quality |
| ROC AUC | 0.96 | Strong overall discrimination |
| Train Time | <0.1s | Enables rapid retraining on new data |
| Interpretability | High | Rules exportable to SIEM systems |

**Alternative:** Logistic Regression — when maximizing Recall (catching every threat) is more important than reducing false alarms.

**Not Recommended:** KNN — despite highest AUC, fails at actual detection (F2 = 0.18).

---

## 6. Limitations

1. **Anonymization**: LANL data is de-identified. Real-world features like IP geolocation, user-agent strings, and application names would likely improve detection.

2. **Extreme Class Imbalance**: Even with SMOTE, the ~0.1% positive rate makes evaluation challenging. Precision remains low across all models.

3. **Sampling**: Results are based on a stratified sample (~638K rows) of the full 1.6B-row dataset. Performance may vary at full scale.

4. **Static Labels**: Red team labels represent only known compromises. There may be additional undetected malicious activity.

5. **Threshold Sensitivity**: F2-Score and confusion matrix results depend on the default 0.5 threshold. Production deployment may require threshold tuning.

---

## 7. Future Work

1. **Threshold Optimization**: Tune classification threshold to maximize F2-Score for specific operational requirements.

2. **Scale Up**: Process the full 1.6B-event dataset using Dask or Spark for distributed computation.

3. **Ensemble Methods**: Experiment with Gradient Boosting (XGBoost, LightGBM) which often outperform single Decision Trees.

4. **Multi-Source Fusion**: Incorporate network flow data (`flows.txt.gz`), DNS lookups (`dns.txt.gz`), and process events (`proc.txt.gz`) for richer behavioral profiles.

5. **Deep Learning**: Explore LSTM autoencoders for sequence-based anomaly detection on per-user event streams.

6. **Online Learning**: Adapt models for streaming/real-time detection as new authentication events arrive.

---

## 8. Conclusion

This project demonstrates that machine learning can effectively detect anomalous authentication behavior in enterprise security logs. The key success factors were:

1. **Feature Engineering**: Transforming raw logs into behavioral metrics (rolling counts, velocities, diversity measures) was more impactful than model selection.

2. **Dual-Metric Evaluation**: Using both ROC AUC and F2-Score revealed that high ranking ability (AUC) does not guarantee good detection (F2). This insight prevented selecting KNN, which would have failed in production.

3. **Interpretability for Production**: The Decision Tree's ability to export human-readable rules bridges the gap between ML experimentation and deployable SIEM logic.

The recommended Decision Tree model achieves a 0.96 ROC AUC and 0.38 F2-Score while training in under 0.1 seconds — enabling rapid adaptation to evolving attack patterns while providing actionable detection rules for security analysts.

---

## References

1. A. D. Kent, "Comprehensive, Multi-Source Cybersecurity Events," Los Alamos National Laboratory, 2015. http://dx.doi.org/10.17021/1179829

2. A. D. Kent, "Cybersecurity Data Sources for Dynamic Network Research," in *Dynamic Networks in Cybersecurity*, Imperial College Press, 2015.

---

## Appendix: Project Structure

```
MLAI-Capstone-Project/
├── README.md                              # Project overview and quick start
├── FINAL_ANALYSIS_REPORT.md               # This report
├── project_problem_statement.md           # Original problem statement
├── requirements.txt                       # Python dependencies
├── notebooks/
│   └── capstone_analysis.ipynb            # Complete analysis notebook
├── src/                                   # Reusable Python modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── clustering_pca.py
│   ├── time_series.py
│   ├── model_selection.py
│   ├── classification.py
│   ├── decision_trees.py
│   └── utils.py
├── data/
│   ├── raw/                               # LANL dataset files
│   └── processed/                         # Generated parquet files
├── images/                                # Saved plot images
└── models/                                # Saved trained models
```
