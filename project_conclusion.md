# Project Conclusion: Anomalous Authentication Behavior Detection

**By Daisy Wu**

---

## 1. Project Overview
For my capstone project, I set out to apply machine learning to a classic cybersecurity problem: detecting anomalous "red team" (compromise) events within enterprise authentication logs. Coming from a decade of software engineering, I'm comfortable with data pipelines and system architecture, but building predictive models from raw data was a new frontier for me. I utilized the Los Alamos National Laboratory (LANL) Cyber Security Dataset to see if ML could effectively isolate these rare events.

## 2. Data Engineering and the Reality of Class Imbalance
My initial data pipeline processed a stratified sample of the dataset, yielding ~638,500 log events. The immediate challenge became apparent: only 702 of these were labeled as malicious. This represents an extreme class imbalance of roughly 0.1%. 

From an engineering perspective, feeding raw log strings (like `user@domain`) directly into a model is ineffective. The real work was in feature engineering—transforming raw logs into behavioral metrics. The most discriminative features turned out to be:
1. `unique_dst_computers_24h`: The count of distinct destination computers a user accessed in a rolling 24-hour window. (This proved to be the most critical indicator of lateral movement).
2. `hour`: The time of day the event occurred.
3. `login_count_24h`: The total volume of logins for a user over 24 hours.

## 3. Model Evaluation
I evaluated three different classification algorithms. Because of the 99.9% class imbalance, standard accuracy is a misleading metric—a model that hardcodes a "normal" response achieves 99.9% accuracy but is useless for security. Therefore, I optimized for ROC AUC and Recall.

*   **K-Nearest Neighbors (KNN)**: Achieved 99.90% accuracy, but failed fundamentally at the actual task, yielding only 14.29% recall and an AUC of 0.8088. It essentially defaulted to predicting the majority class.
*   **Logistic Regression**: Showed significant improvement. By applying balanced class weights, it achieved an AUC of 0.9529 and a recall of 88.57%. The overall accuracy dropped to 89.01%, which is an acceptable trade-off for catching the minority class.
*   **Decision Tree**: Emerged as the strongest performer. It achieved the highest AUC score of 0.9571, with a solid recall of 83.57% and an accuracy of 91.74%. 

## 4. Architectural Decision: The Best Model
I selected the **Decision Tree** (tuned to a max depth of 7) as the optimal model for this use case. 

Beyond the raw metrics (highest ROC AUC), the Decision Tree offers a crucial advantage for production systems: interpretability. Unlike the opaque coefficients of Logistic Regression or the distance metrics of KNN, a Decision Tree outputs explicit boolean logic. These rules (e.g., `if unique_dst_computers_24h > X and hour < Y`) can be directly exported and implemented as deterministic alerts in an existing SIEM (Security Information and Event Management) system, bridging the gap between an ML experiment and a deployable software solution.

## 5. Key Takeaways
Transitioning from deterministic software engineering to probabilistic machine learning highlighted that the algorithm is only as good as the data representation. The success of this project wasn't driven by choosing a complex model, but by engineering stateful, rolling-window features that accurately captured user behavior. Furthermore, handling edge cases in ML—like extreme class imbalance—requires tuning the evaluation metrics (AUC/Recall) rather than relying on standard accuracy.
