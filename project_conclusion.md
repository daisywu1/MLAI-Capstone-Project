# Project Conclusion: Anomalous Authentication Behavior Detection

**AI/ML Capstone Project — Daisy Wu**

---

## 1. Problem Statement Recap

This project set out to determine whether machine learning techniques can effectively identify and classify anomalous authentication behavior in enterprise security logs. Using the Los Alamos National Laboratory (LANL) Cyber Security Dataset — containing ~1.6 billion authentication events over 58 days with 749 known red team compromise events — we built a complete ML pipeline from raw log ingestion through model comparison and interpretable rule extraction.

## 2. Methodology Summary

### Data Pipeline

- **Source**: LANL `auth.txt.gz` (~7.2 GB compressed, ~1.6B rows) and `redteam.txt.gz` (749 labeled compromise events)
- **Sampling Strategy**: Stratified random sampling across the full 58-day time range, producing ~1M representative events that span the entire dataset period
- **Preprocessing**: Parsed `user@domain` fields, converted outcomes to binary, derived temporal features (hour, day, weekend, off-hours flags), and matched red team labels to authentication events

### Feature Engineering (Module 8)

We constructed 16 behavioral features from raw log fields:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Temporal** | hour, day_of_week, is_off_hours, is_weekend | Attackers often operate outside business hours |
| **Frequency** | login_count_1h, login_count_6h, login_count_24h | Brute-force and credential stuffing produce bursts |
| **Failure** | fail_count_1h, fail_ratio_24h | Failed authentication attempts signal compromise attempts |
| **Diversity** | unique_dst_computers_24h, unique_auth_types_24h, unique_logon_types_24h | Lateral movement touches many machines |
| **Behavioral** | time_since_last_login, is_new_dst_computer, events_per_minute_10m | Compromised accounts show unusual access patterns |

### Techniques Applied

| Technique | Module | Application |
|-----------|--------|-------------|
| PCA & K-Means Clustering | Module 6 | Reduced 16 features to 2 principal components for visualization; identified 3 natural behavioral clusters |
| Feature Engineering & Overfitting | Module 8 | Built behavioral features with train/test splits and cross-validation |
| Model Selection & Regularization | Module 9 | Compared 4 models via 5-fold CV; tuned L1/L2 regularization strength |
| Time Series Analysis | Module 10 | Decomposed hourly event volume into trend/seasonal/residual; detected temporal anomalies via rolling z-score |
| KNN Classification | Module 11 | Tuned k from 1–15 via cross-validation |
| Logistic Regression | Module 13 | Baseline classifier with interpretable feature weights |
| Decision Trees | Module 14 | Produced human-readable rules for security analysts |
| Gradient Descent & Optimization | Module 15 | Reflected in solver selection and convergence tuning for logistic regression |

## 3. Key Findings

### 3.1 Exploratory Data Analysis

- **Volume**: The dataset shows strong daily seasonality with peak activity during business hours and reduced volume overnight and on weekends.
- **Success Rate**: >99% of authentication events succeed; failures are rare but concentrated among specific users and computers.
- **Authentication Types**: Kerberos dominates (~86%), followed by NTLM (~10%) and Negotiate (~3%). Red team events use a mix of types.
- **Class Imbalance**: Red team events represent ~0.00005% of all events — an extreme imbalance that requires careful handling.

### 3.2 PCA and Clustering

- **Variance**: The first 2 principal components capture ~49% of variance; 8 components reach ~95%.
- **Clusters**: K-Means with K=3 (optimal by silhouette score = 0.79) reveals:
  - **Cluster 0** (largest): Normal automated/service account activity
  - **Cluster 1**: Normal interactive user sessions
  - **Cluster 2** (smallest): Unusual or high-frequency events — the cluster most likely to contain anomalies

### 3.3 Time Series Analysis

- **Seasonality**: Clear 24-hour periodicity in event volume, with business-hour peaks and overnight troughs.
- **Anomalies**: Rolling z-score detection (window=24h, threshold=3σ) flags time bins with unusual spikes or dips that may correlate with red team activity or system events.

### 3.4 Model Comparison

All models were evaluated using 5-fold cross-validation with ROC AUC as the primary metric, given the extreme class imbalance:

| Model | Strengths | Limitations |
|-------|-----------|-------------|
| **Logistic Regression (L2)** | Interpretable coefficients, fast training, strong baseline | Linear decision boundary |
| **Logistic Regression (L1)** | Feature selection via coefficient sparsity | Slightly lower AUC than L2 |
| **KNN** | Captures local behavioral patterns | Slow inference on large datasets |
| **Decision Tree** | Produces actionable rules, handles nonlinearity | Prone to overfitting without depth control |

### 3.5 Most Important Features

Across all models, the most discriminative features were:

1. **login_count_1h** — High-frequency login bursts are a strong indicator
2. **events_per_minute_10m** — Velocity metric captures automated attack tools
3. **is_new_dst_computer** — Lateral movement to previously unseen machines
4. **time_since_last_login** — Dormant accounts suddenly becoming active
5. **fail_ratio_24h** — Elevated failure rates signal credential guessing

### 3.6 Interpretable Rules for Security Analysts

The decision tree produces rules that can be directly deployed as SIEM alert logic:

- *"If `login_count_1h` > X AND `is_new_dst_computer` = 1 AND `is_off_hours` = 1 → Suspicious"*
- *"If `events_per_minute_10m` > Y AND `fail_ratio_24h` > Z → Suspicious"*

These provide immediate detection capability without requiring ML model deployment in production.

## 4. Limitations

1. **Anonymized Data**: The LANL dataset is de-identified — real-world features like IP geolocation, user-agent strings, and application names are unavailable. These would likely improve detection significantly.

2. **Extreme Class Imbalance**: With only 749 positive events out of ~1.6 billion, standard accuracy is meaningless. We rely on ROC AUC, precision-recall, and stratified evaluation, but the rarity of positives limits what supervised models can learn.

3. **Sampling**: We analyzed ~1M rows (a stratified sample). Results may shift at full scale, particularly for rare-event detection.

4. **Label Completeness**: Red team labels represent *known* compromises only. There may be additional undetected malicious activity that our models could surface but cannot be validated.

5. **Static Analysis**: This is a batch analysis. Real-world deployment would require streaming/online learning to detect threats in real time.

## 5. Future Work

- **Full-Scale Processing**: Use Dask or Apache Spark to process all 1.6 billion events.
- **Ensemble Methods**: Experiment with XGBoost, LightGBM, and Random Forests for improved detection.
- **Multi-Source Fusion**: Incorporate LANL's network flow (`flows.txt.gz`), DNS (`dns.txt.gz`), and process (`proc.txt.gz`) data for richer behavioral profiles.
- **Deep Learning**: Explore LSTM autoencoders for sequence-based anomaly detection on per-user event streams.
- **Online Learning**: Adapt models for streaming detection as new events arrive.
- **SMOTE / Oversampling**: Apply synthetic minority oversampling to address the extreme class imbalance.

## 6. Conclusion

This project demonstrates that machine learning can effectively surface anomalous authentication behavior in enterprise security logs, even with extreme class imbalance and anonymized data. The combination of behavioral feature engineering, unsupervised clustering, and supervised classification provides multiple complementary detection angles. Most importantly, the decision tree approach produces interpretable rules that security analysts can immediately deploy — bridging the gap between ML research and operational security.

## References

1. A. D. Kent, "Comprehensive, Multi-Source Cybersecurity Events," Los Alamos National Laboratory, 2015. http://dx.doi.org/10.17021/1179829

2. A. D. Kent, "Cybersecurity Data Sources for Dynamic Network Research," in *Dynamic Networks in Cybersecurity*, Imperial College Press, 2015.
