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

### Supervised Learning Models & Hyperparameter Tuning
Hyperparameters were tuned via `GridSearchCV` using 5-fold cross-validation:
| Model | Parameter Grid | Rationale |
|-------|----------------|-----------|
| Logistic Regression / SVM | `C`: [0.01, 0.1, 1, 10] | `C` controls regularization. Smaller values enforce stronger regularization to prevent overfitting on noisy logs, while larger values allow closer data fitting. |
| Decision Tree | `max_depth`: [3, 5, 7, 10] | Balances interpretability (shallow trees) against capturing complex behavioral patterns (deeper trees). |
| Random Forest | `n_estimators`: [50, 100], `max_depth`: [5, 10] | `n_estimators` balances ensemble robustness against training elapsed time. `max_depth` limits individual tree complexity. |

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

### Model Comparison (True Test Set Performance)

| Model | ROC AUC | F2-Score | Recall | Precision | Train Time |
|-------|---------|----------|--------|-----------|------------|
| **Baseline (Random)** | ~0.50 | ~0.00 | ~0.50 | <0.01 | <0.01s |
| K-Nearest Neighbors | 0.8462 | **0.2771** | 0.6500 | **0.0841** | **~0.02s** |
| Random Forest | **0.9786** | 0.1849 | 0.8357 | 0.0449 | ~12.2s |
| Logistic Regression | 0.9522 | 0.0434 | **0.8786** | 0.0090 | ~0.8s |
| Decision Tree | 0.9275 | 0.1478 | 0.7214 | 0.0354 | ~3.7s |
| Support Vector Machine| N/A | 0.0429 | 0.8786 | 0.0089 | ~1.1s |

### Key Observations

1. **Test Set vs. Cross-Validation**: Evaluating on a pristine test set (unaffected by SMOTE over-sampling) reveals the true difficulty of the extreme class imbalance (0.1%). Linear models like Logistic Regression achieve great Recall (88%) but abysmal Precision (<1%).

2. **KNN has the best F2-Score (0.277)**. It strikes the best out-of-the-box balance by maintaining the highest Precision (8.4%), meaning it limits the false alarm rate to an acceptable level while catching 65% of threats.

3. **Random Forest is the most robust overall**: Highest AUC (0.978) and strong Recall (83.5%). Its ranking capability is unmatched, meaning its threshold can be easily tuned to optimize the F2-Score for production.

4. **Training Elapsed Time vs Robustness**: The time it takes to train a model heavily impacts its viability in production. **KNN** takes just `~0.02s` and **Decision Tree** takes `~3.7s`, making them ideal for rapid, daily retraining. While **Random Forest** provides unmatched robustness (AUC 0.978), it requires significantly more training time (`~12.2s`), introducing a trade-off between speed and capability.

5. **Interpretability**: Decision Trees provide a "good enough" baseline while producing readable boolean rules, but sacrifice too much predictive power compared to the Random Forest ensemble.

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

**Selected Model: Random Forest (n_estimators=100, max_depth=10)**

| Criterion | Value | Why it matters |
|-----------|-------|----------------|
| ROC AUC | 0.9786 | Best overall ability to rank threats above normal events |
| Recall | 83.57% | Catches the vast majority of malicious behavior |
| F2-Score | 0.1849 | Good baseline detection quality, highly tunable |

**Alternative:** K-Nearest Neighbors (KNN) — Recommended if the SOC requires immediate out-of-the-box deployment with minimal false alarms (highest precision of 8.4%), and is willing to accept a lower recall (65%).

**Not Recommended:** Logistic Regression / SVM — Despite catching nearly 88% of threats, their <1% precision means they would flood security analysts with over 100 false alarms for every real threat.

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

1. **Feature Engineering**: Transforming raw logs into behavioral metrics (rolling counts, velocities, diversity measures) was far more impactful than algorithm selection alone.
2. **True Test Set Evaluation**: The extreme class imbalance (0.1%) makes cross-validation tricky when using over-sampling techniques like SMOTE. Evaluating strictly on a pristine test set revealed the true operational capabilities (Precision/Recall trade-off) of each model.
3. **Dual-Metric Evaluation**: Using both ROC AUC and F2-Score highlighted that linear models catching 88% of threats were actually unusable due to <1% precision. The **Random Forest** emerged as the most robust ranking model (97.8% AUC), while **KNN** provided the best out-of-the-box F2-Score.

The recommended Random Forest model provides a robust foundation for automated threat detection, capable of discovering previously unseen patterns while providing the tunability needed to integrate into modern Security Operations Centers.

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
