# Fake News Detection on WELFake (Classical Machine Learning + SHAP for Logistic Regression & XGBoost)

This directory contains the **classical machine-learning pipeline** for detecting fake news on the **WELFake** dataset.  
The notebook `welfake_ml.ipynb` demonstrates end-to-end preprocessing, TF–IDF feature engineering, model training, evaluation, and **SHAP interpretability** for two key models:  
**Logistic Regression** and **XGBoost**.

---

## Project Overview

The notebook implements the full classical ML workflow:

- Load and clean the **WELFake_Dataset.csv** dataset  
- Preprocess text (lowercasing, punctuation removal, join title + content)  
- Convert text into TF–IDF vector features  
- Train and evaluate a suite of classical ML models:
  - **Logistic Regression**
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Machine (SVM, linear kernel)**
  - **Gaussian Naive Bayes**
  - **Decision Tree**
  - **Random Forest**
  - **XGBoost**
- Apply **SHAP interpretability** to understand feature contributions:
  - **Logistic Regression** → LinearExplainer / KernelExplainer
  - **XGBoost** → TreeExplainer with `pred_contribs=True`
- Visualize interpretable outputs:
  - **SHAP summary plots** (global feature importance)
  - **Local explanations** for individual predictions  

---

## Why These Models?

### **Logistic Regression**
- Strong and reliable baseline for text classification  
- Linear decision boundary makes it naturally interpretable  
- SHAP values closely match model coefficients  

### **XGBoost**
- Large set of boosted decision trees  
- Captures nonlinear interactions between words  
- SHAP (TreeSHAP) provides **exact, fast** explanations  

### Other Models Included
KNN, Naive Bayes, Decision Tree, and Random Forest are evaluated for comparison,  
but **SHAP is not applied to them** due to:
- Higher computational cost  
- Lower interpretability gain  
- Compatibility limitations (especially for Random Forest SHAP)  

---

## SHAP Interpretability

The ML section uses SHAP to provide a transparent understanding of predictions.

### **Logistic Regression**
- Uses **LinearExplainer** when available  
- Or **KernelExplainer** as a fallback  
- Shows which TF–IDF tokens push predictions toward “FAKE” or “REAL”

### **XGBoost**
- Uses **TreeExplainer** with exact SHAP computation  
- Fast, highly accurate feature attributions  
- Displays token-level contributions across thousands of trees  

### Outputs include:
- **Summary plots** (global)
- **Force plots** (local)
- **Token importance ranking**
- **Analysis of linguistic patterns used by the model**

---

## Running the Models

Ensure the dataset is stored in the same directory or adjust the path:

```python
df = pd.read_csv("WELFake_Dataset.csv")
