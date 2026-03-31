# Capstone Problem Statement

## Overview of the Question

This project aims to determine whether machine learning techniques can effectively identify and classify anomalous authentication behavior in enterprise security logs. By analyzing historical access data, the goal is to build a predictive model that distinguishes normal user login patterns from potentially suspicious activity — such as credential stuffing, brute-force attempts, or compromised account usage — enabling faster detection and response.

## Type of Data Needed

The project will use authentication log data typically available in cybersecurity environments, including:

- **Raw fields:** timestamps, user identifiers, source IP addresses, geographic location (country/region), device or browser metadata (user-agent strings), authentication method, login success/failure indicators, and target resource or application.  
- **Engineered features:** login frequency per user over rolling time windows, failed-to-successful login ratios, number of distinct IPs or geolocations per user per day, time since last login, hour-of-day and day-of-week indicators, and velocity metrics (e.g., logins per minute). Feature engineering and careful handling of overfitting (Module 8 will be central to this phase.

Publicly available datasets such as the **Los Alamos National Laboratory (LANL) Cyber Security Dataset** or synthetic authentication logs can serve as a starting point if enterprise data is not available.

## Proposed Techniques


| Technique                                                           | Module          | Application to This Project                                                                                                                                                                                                                             |
| ------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Clustering and PCA**                                              | Module 6        | Use PCA to reduce the high-dimensional engineered feature space for visualization and noise reduction. Apply clustering (K-Means) to discover natural groupings of login behavior — potentially surfacing anomalous clusters without relying on labels. |
| **Feature Engineering and Overfitting**                             | Module 8        | Engineer meaningful behavioral features from raw log fields. Use train/test splits and cross-validation to ensure features generalize and the model does not overfit to historical patterns.                                                            |
| **Model Selection and Regularization**                              | Module 9        | Compare multiple candidate models using cross-validation. Apply regularization (Ridge/Lasso) to logistic regression to prevent overfitting on high-dimensional feature sets and to identify the most important predictors.                              |
| **Time Series Analysis**                                            | Module 10       | Model temporal trends in user login behavior (e.g., seasonal patterns, gradual drift) and detect sudden deviations that may indicate compromise.                                                                                                        |
| **Classification with k-Nearest Neighbors and Logistic Regression** | Modules 11 & 13 | Build baseline classifiers to label login events as normal or suspicious. Logistic regression provides interpretable feature weights; KNN captures local behavioral similarities.                                                                       |
| **Decision Trees**                                                  | Module 14       | Model nonlinear relationships and produce interpretable rule sets (e.g., "if failed logins 5 AND new IP AND off-hours, flag as suspicious") that security analysts can act on directly.                                                                 |
| **Gradient Descent and Optimization**                               | Module 15       | Understand and tune the optimization process underlying logistic regression and other models to improve convergence and performance.                                                                                                                    |


