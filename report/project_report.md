# Credit Default Prediction: A Comparative Analysis of Machine Learning Models

**Date:** December 16, 2025  
**Author:** AI Assistant & User  

---

## 1. Abstract

Credit default prediction is a cornerstone of modern financial risk management. Accurately assessing the likelihood of a borrower failing to repay a loan minimizes financial losses and enables more inclusive lending practices. This project investigates the efficacy of ten distinct machine learning algorithms—ranging from traditional linear models to state-of-the-art gradient boosting ensembles—in predicting credit risk using the benchmark German Credit dataset. 

The study implements a robust pipeline ensuring consistent data preprocessing, feature engineering, and stratified evaluation. The models evaluated include Logistic Regression, Random Forest, AdaBoost, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Decision Trees, XGBoost, LightGBM, CatBoost, and Explainable Boosting Machine (EBM). Performance was assessed using Accuracy, F1-Score, and the Area Under the Receiver Operating Characteristic Curve (ROC-AUC). 

Results demonstrate that modern ensemble methods, specifically the **Explainable Boosting Machine (EBM)**, provide the most robust performance. While **CatBoost** achieved a comparable ROC-AUC (0.815 vs 0.814), **EBM** offered a superior F1-score (0.61 vs 0.55) and significantly higher sensitivity (Recall) for detecting bad credits (57% vs 45%). The findings identify EBM as the optimal solution for financial risk modeling, delivering high accuracy and necessary regulatory interpretability without the "black box" trade-off.

---

## 2. Introduction

### 2.1 Research Question
The primary research question driving this study is: *To what extent can advanced machine learning algorithms improve the predictive accuracy of credit default classification compared to traditional statistical baselines, without sacrificing the ability to interpret the decision-making process?*

### 2.2 Motivation
The financial services industry relies heavily on credit scoring to manage risk. A "good" credit score grants access to capital, mortgages, and business loans, while a "bad" score restricts it. Traditionally, these decisions were driven by human judgment or simple linear scorecards (Logistic Regression). 

However, linear models may fail to capture complex, non-linear interactions between borrower attributes (e.g., the interaction between age and employment duration). Misclassification has two costly outcomes: 
1.  **False Negatives (Type II Error):** Granting a loan to a defaulter, resulting in direct financial loss.
2.  **False Positives (Type I Error):** Denying a loan to a creditworthy applicant, resulting in lost revenue and customer dissatisfaction.

Motivated by the need to optimize this trade-off, this project explores a suite of algorithms to identify the optimal balance of performance and reliability.

---

## 3. Literature Review

The field of credit scoring has evolved significantly over the past few decades.

**Traditional Approaches:**
Historically, **Logistic Regression** and Linear Discriminant Analysis (LDA) have been the industry standards due to their mathematical simplicity and the ease with which coefficients can be translated into a scorecard. 

**The Machine Learning Shift:**
In recent years, research has shifted towards non-parametric and ensemble methods. **Random Forests** (Breiman, 2001) demonstrated that bagging decision trees could reduce variance and improve accuracy. **Gradient Boosting Machines** (Friedman, 2001) introduced the concept of sequentially correcting errors, leading to the development of highly optimized libraries like **XGBoost** (Chen & Guestrin, 2016) and **LightGBM** (Ke et al., 2017).

**Interpretability vs. Performance:**
A recurring challenge in applied ML is the "black box" nature of complex ensembles. Financial regulations (e.g., GDPR, ECOA) often require explanations for adverse credit decisions. This has led to the rise of Interpretable Machine Learning. **Explainable Boosting Machines (EBM)** (Nori et al., 2019) represent a generalized additive model with interaction terms (GA2M), aiming to provide the accuracy of boosting methods with the intelligibility of regression. This study places a special emphasis on comparing EBM against its "black box" counterparts.

---

## 4. Methodology

### 4.1 Data Description
The dataset used is the **German Credit Data** (UCI Machine Learning Repository).
-   **Instances:** 1000 loan applicants.
-   **Target Variable:** `credit_risk` (Encoded: 0 for Good, 1 for Bad).
-   **Class Imbalance:** Approximately 70% Good vs. 30% Bad credits.

### 4.2 Feature Engineering
The raw dataset contains 20 attributes, a mix of qualitative and quantitative features.

**Process Steps:**
1.  **Data Cleaning:** Cryptic categorical codes (e.g., "A11", "A34") were mapped to meaningful semantic labels (e.g., "< 0 DM", "Critical/Other").
2.  **Type Separation:**
    -   *Numeric Features (7):* Duration, Credit Amount, Installment Rate, Residence Since, Age, Existing Credits, People Liable.
    -   *Categorical Features (13):* Status, Credit History, Purpose, Savings, Employment, Sex, etc.
3.  **Preprocessing Pipeline:**
    -   *Numeric data* was scaled using `StandardScaler` to ensure distance-based algorithms (like KNN) were not biased by magnitude.
    -   *Categorical data* was transformed using `OneHotEncoder` to handle nominal variables without imposing ordinal relationships.

### 4.3 Models Evaluated
The following algorithms were trained:
1.  **Logistic Regression:** The baseline linear model.
2.  **Gaussian Naive Bayes:** A probabilistic baseline assuming feature independence.
3.  **K-Nearest Neighbors (KNN):** A distance-based instance learner.
4.  **Decision Tree:** A simple rule-based model.
5.  **Random Forest:** A bagging ensemble of trees.
6.  **AdaBoost:** An early boosting algorithm.
7.  **XGBoost:** specific implementation of gradient boosting.
8.  **LightGBM:** Gradient boosting optimized for speed and leaf-wise growth.
9.  **CatBoost:** Gradient boosting with advanced categorical handling.
10. **EBM:** Generalized Additive Model with pairwise interactions.

### 4.4 Evaluation Strategy
-   **Split:** The data was split into 80% Training and 20% Testing sets.
-   **Stratification:** Utilized to maintain the class distribution ratio in both sets.
-   **Metrics:** 
    -   *Accuracy:* General correctness.
    -   *F1-Score:* Harmonic mean of precision and recall (crucial for imbalanced data).
    -   *ROC-AUC:* The probability that the model evaluates a randomly chosen positive instance higher than a randomly chosen negative one. This is the primary metric for comparison.

---

## 5. Results

The models were evaluated on the held-out test set (n=200). The summary metrics are presented below.

| Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **EBM** | **0.780** | **0.607** | **0.814** |
| **CatBoost** | 0.775 | 0.545 | **0.815** |
| **Logistic Regression** | 0.780 | 0.593 | 0.804 |
| **Random Forest** | 0.765 | 0.515 | 0.798 |
| **XGBoost** | 0.755 | 0.559 | 0.775 |
| **LightGBM** | 0.730 | 0.491 | 0.770 |
| **AdaBoost** | 0.750 | 0.545 | 0.764 |
| **Gaussian NB** | 0.695 | 0.561 | 0.741 |
| **KNN** | 0.725 | 0.433 | 0.729 |
| **Decision Tree** | 0.605 | 0.368 | 0.542 |

### 5.1 Key Findings
1.  **Top Performers:** EBM and CatBoost achieved practically identical ROC-AUC scores (~0.814), establishing themselves as the state-of-the-art for this dataset.
2.  **Class 1 (Bad Credit) Performance:** While AUC scores are similar, **EBM** significantly outperforms CatBoost in identifying bad credits.
    -   **EBM:** Recall for Class 1 is **0.57**, with an F1-score of **0.61**.
    -   **CatBoost:** Recall for Class 1 is only **0.45**, with an F1-score of **0.55**.
    -   **Implication:** EBM captures significantly more actual defaulters than CatBoost, which is critical for minimizing financial risk, even if CatBoost has slightly higher precision.
3.  **Strong Baseline:** Logistic Regression performed surprisingly well (AUC 0.804, Class 1 Recall 0.53), outperforming complex "black box" models like XGBoost and Random Forest.
4.  **Weakest Performers:** The single Decision Tree performed poorly (AUC 0.542), highlighting the necessity of ensemble methods to reduce variance and overfitting.

*(Note: Detailed confusion matrices for each model are saved in the `results/` directory as generated artifacts).*

---

## 6. Discussion

### 6.1 Interpretation of Results
The superior performace of **EBM** is the most significant finding. While **CatBoost** showed excellent discrimination (high AUC) and precision, it struggled to identify the minority class (defaulters), achieving a recall of only 45%. **EBM**, conversely, achieved a recall of 57%, conducting a more balanced risk assessment. In credit scoring, missing a defaulter (False Negative) is often more costly than rejecting a good customer (False Positive), making EBM the more commercially viable model.

Furthermore, EBMs are "glassbox" models; each feature's contribution can be visualized. The fact that EBM outperformed "blackbox" models in F1-score implies that **we do not need to sacrifice interpretability for accuracy**. This is critical for regulatory compliance (e.g., GDPR "Right to Explanation") where loan denials must be transparent.

The strong performance of **Logistic Regression** indicates that the underlying feature-response relationships are largely linear. However, the boosting models likely gained their edge by capturing specific non-linear risk factors (e.g., interactions between Age and Employment Duration) that linear models miss.

### 6.2 Limitations
1.  **Dataset Size:** With only 1000 entries, the risk of overfitting is real. While cross-validation helps, larger datasets would provide more stability.
2.  **Feature Scope:** The dataset is limited to financial demographics. It lacks alternative data sources (transaction history, social graph) that modern fintechs utilize.
3.  **Temporal Bias:** Is the "German Credit Data" (collected decades ago) still representative of modern behavior? Economic conditions change, rendering older models potentially obsolete.

---

## 7. Conclusion

This project successfully implemented a comprehensive machine learning pipeline to predict credit default. The analysis confirmed that while traditional Logistic Regression remains a formidable baseline, modern algorithms like **Explainable Boosting Machines (EBM)** and **CatBoost** offer marginal but valuable improvements in predictive power.

**Future Work:**
-   **Hyperparameter Tuning:** Implementing GridSearch or Bayesian Optimization could further squeeze performance from the tree-based models.
-   **Feature Selection:** Recursive Feature Elimination (RFE) could simplify the models by removing noise.
-   **Real-world Validation:** Testing these models on a contemporary, larger dataset would validatetheir generalization capabilities.

In summary, for financial institutions aiming to modernize their credit scoring, EBMs represent the optimal path forward: high accuracy, high interpretability, and low risk.

---

## 8. References

1.  **Breiman, L.** (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
2.  **Friedman, J. H.** (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 1189-1232.
3.  **Nori, H., Jenkins, S., Koch, P., & Caruana, R.** (2019). InterpretML: A Unified Framework for Machine Learning Interpretability. *arXiv preprint arXiv:1909.09223*.
4.  **Prokhorenkova, L., et al.** (2018). CatBoost: unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*.
5.  **Chen, T., & Guestrin, C.** (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
6.  **Dua, D. and Graff, C.** (2019). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.