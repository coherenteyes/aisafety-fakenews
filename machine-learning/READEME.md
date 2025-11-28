# Fake News Detection on WELFake (Classical ML + SHAP for Logistic Regression & XGBoost)

This repository contains a machine-learning pipeline for detecting fake news on the **WELFake** dataset using classical models and interpretable explanations. The notebook `welfake_ml.ipynb` walks through preprocessing, TF-IDF feature construction, training multiple classifiers, and generating **SHAP explanations** for two models:  
**(1) Logistic Regression** and **(2) XGBoost**.

---

## 1. Project Overview

The notebook implements the following steps:

- Load and clean the **WELFake_Dataset.csv** fake-news dataset  
- Convert text into TF-IDF features (title + article body)
- Train and evaluate multiple classical ML models:
  - Logistic Regression  
  - K-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM, linear kernel)  
  - Gaussian Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - XGBoost
- Provide **model interpretability** using:
  - **SHAP for Logistic Regression** (KernelExplainer / LinearExplainer depending on setup)  
  - **SHAP for XGBoost** (TreeExplainer with `pred_contribs=True`)  
- Visualize:
  - Global feature attribution (summary plots)  
  - Local instance-level explanations (sample-level SHAP values)

**Note:**  
SHAP is only applied to **Logistic Regression** and **XGBoost** due to computational constraints and compatibility.  
Decision Tree and Random Forest are included as models but **not interpreted with SHAP** in this version.

---

## 2. Dataset Information

- **Dataset:** WELFake  
- **File expected:** `WELFake_Dataset.csv`  
- **Main columns**
  - `title`
  - `text`
  - `label` (0 = real, 1 = fake)

Place the file in the same folder as the notebook or adjust the path:

```python
df = pd.read_csv("WELFake_Dataset.csv")
