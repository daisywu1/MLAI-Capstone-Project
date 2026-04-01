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

My data pipeline processed a stratified sample of the dataset, yielding ~638,500 log events, of which only 702 were labeled as malicious (an extreme class imbalance of roughly 0.1%).

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
| **Time Series Analysis** | Decomposed hourly event volume to model temporal trends (seasonality) and detected sudden deviations via rolling z-scores. |
| **Model Selection & Regularization** | Compared models using 5-fold cross-validation. Tuned L1/L2 regularization for Logistic Regression to handle high-dimensional spaces and prevent overfitting. |
| **Classification (KNN & Logistic Regression)** | Built baseline classifiers. Logistic Regression provided interpretable feature weights, while KNN attempted to capture local similarities. |
| **Decision Trees** | Modeled nonlinear relationships to produce interpretable, human-readable rule sets for security analysts. |

## 4. Model Evaluation & Conclusion
I evaluated three different classification algorithms. Because of the extreme class imbalance, standard accuracy is a misleading metric—a model that hardcodes a "normal" response achieves 99.9% accuracy but is useless for security. Therefore, I optimized for ROC AUC and Recall.

*   **K-Nearest Neighbors (KNN)**: Achieved 99.90% accuracy, but failed fundamentally at the actual task, yielding only 14.29% recall and an AUC of 0.8088. It essentially defaulted to predicting the majority class.
*   **Logistic Regression**: Showed significant improvement. By applying balanced class weights, it achieved an AUC of 0.9529 and a recall of 88.57%. The overall accuracy dropped to 89.01%, which is an acceptable trade-off for catching the minority class.
*   **Decision Tree**: Emerged as the strongest performer. It achieved the highest AUC score of 0.9571, with a solid recall of 83.57% and an accuracy of 91.74%. 

### Architectural Decision: The Best Model
I selected the **Decision Tree** (tuned to a max depth of 7) as the optimal model for this use case. 

Beyond the raw metrics (highest ROC AUC), the Decision Tree offers a crucial advantage for production systems: interpretability. Unlike the opaque coefficients of Logistic Regression or the distance metrics of KNN, a Decision Tree outputs explicit boolean logic. These rules (e.g., `if unique_dst_computers_24h > X and hour < Y`) can be directly exported and implemented as deterministic alerts in an existing SIEM (Security Information and Event Management) system, bridging the gap between an ML experiment and a deployable software solution.

### Key Takeaways
Transitioning from deterministic software engineering to probabilistic machine learning highlighted that the algorithm is only as good as the data representation. The success of this project wasn't driven by choosing a complex model, but by engineering stateful, rolling-window features that accurately captured user behavior. Furthermore, handling edge cases in ML—like extreme class imbalance—requires tuning the evaluation metrics (AUC/Recall) rather than relying on standard accuracy.

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