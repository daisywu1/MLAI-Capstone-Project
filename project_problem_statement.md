# AI/ML Capstone Project Problem Statement
**Author:** Daisy Wu, Software Engineer

## 1. The Research Question
Can machine learning techniques effectively identify and classify anomalous authentication behavior within massive enterprise security logs to accurately detect compromised accounts and lateral movement?

## 2. Expected Data Source(s)
This project utilizes the **Los Alamos National Laboratory (LANL) Cyber Security Dataset**, a comprehensive collection of de-identified enterprise network events. The dataset encompasses 58 consecutive days of authentication logs, representing approximately 1.6 billion individual events. Crucially, it includes labels for 749 rare, known malicious "red-team" compromise events, providing the ground truth necessary for supervised learning. 

**Data Source Link:** [https://csr.lanl.gov/data/cyber1/](https://csr.lanl.gov/data/cyber1/)

## 3. Techniques Used
To process the data and build a highly effective detection pipeline, the following machine learning and data science techniques were employed:
*   **Exploratory Data Analysis (EDA) & Time Series Analysis:** To understand login distributions, daily seasonality, and detect temporal anomalies via rolling z-scores.
*   **Feature Engineering:** Transforming raw log fields into powerful behavioral metrics, such as `unique_dst_computers_24h` (to detect lateral movement) and `login_count_24h`.
*   **SMOTE (Synthetic Minority Over-sampling Technique):** Applied to the training data to synthesize rare red-team events and counteract the extreme class imbalance (~0.00005% positive).
*   **PCA & K-Means Clustering:** Used for dimensionality reduction and to discover natural behavioral groupings without relying on labels.
*   **Supervised Learning:** Training and evaluating multiple classifiers including K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machines (SVM), Random Forest, and Decision Trees.
*   **Model Selection & Interpretability:** Utilizing `GridSearchCV` for hyperparameter tuning and extracting human-readable rule sets and SHAP summary plots to explain model predictions.

## 4. Expected Results
Because the dataset exhibits an extreme class imbalance, standard accuracy is a misleading metric (predicting "normal" 100% of the time yields 99.9% accuracy but fails to detect threats). Therefore, the project expects to optimize for and evaluate models based on **ROC AUC and Recall**.

The anticipated optimal model is the **Decision Tree**, expected to achieve a high ROC AUC (0.9571) and strong recall. Beyond the statistical metrics, the most critical expected result is the ability to extract deterministic rule sets—translating probabilistic ML predictions into explicit boolean logic (e.g., `if unique_dst_computers_24h > X and hour < Y`). This bridges the gap between theoretical data science and practical cybersecurity application.

## 5. Importance of the Question
In today's digital landscape, the cost of a security breach can be catastrophic, leading to millions in financial losses, severe regulatory fines, and irreparable reputational damage. When an attacker compromises an account, the speed of response dictates the blast radius of the breach.

Relying on manual analysis or traditional threshold-based alerting is impossible when dealing with billions of authentication events. Machine learning provides the capability to sift through this massive volume of data to find the "needle in the haystack." 

However, security operation centers (SOCs) cannot effectively act on opaque, "black-box" AI predictions. By transitioning the probabilistic findings of an AI model into actionable, deterministic logic (like Decision Tree rules), we empower non-data science security teams. This approach allows security engineers to fully understand, trust, and seamlessly deploy the detection logic directly into their existing SIEM (Security Information and Event Management) infrastructure, ultimately accelerating threat isolation and reducing the overall cost of a breach.