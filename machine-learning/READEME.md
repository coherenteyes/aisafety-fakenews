# Fake News Detection on WELFake (Classical ML + SHAP)

This repository contains a notebook for training and interpreting classical machine-learning models on the **WELFake** fake-news dataset. The pipeline covers text preprocessing, TF-IDF feature extraction, multiple ML baselines, and **SHAP-based model interpretability** for tree ensembles.

---

## 1. Project Overview

The notebook `welfake_ml.ipynb`:

- Loads the **WELFake_Dataset.csv** fake-news dataset (balanced real vs fake news).
- Cleans and tokenizes article **titles** and **bodies**.
- Builds TF-IDF features (title + text).
- Trains and evaluates multiple classifiers:
  - Logistic Regression  
  - k-Nearest Neighbors (KNN)  
  - Support Vector Machine (SVM, linear kernel)  
  - Gaussian Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - XGBoost
- Uses **SHAP (SHapley Additive exPlanations)** to:
  - Visualize **global** feature importance (summary plots).
  - Inspect **local** explanations for individual predictions (waterfall plots).

This gives both performance benchmarks and interpretability for fake-news detection on a modern, large-scale dataset.

---

## 2. Dataset

- **Name:** WELFake Dataset  
- **File expected by the notebook:** `WELFake_Dataset.csv`  
- **Columns used:**
  - `title`: news headline  
  - `text`: full news article  
  - `label`: ground truth (e.g., 0 = real, 1 = fake)

> Make sure the CSV is in the **same folder** as `welfake_ml.ipynb` or update the path in the cell:
> ```python
> df = pd.read_csv('WELFake_Dataset.csv')
> ```

---

## 3. Methods

### 3.1 Text Preprocessing

Main steps in the notebook:

1. **Basic cleaning**
   - Remove punctuation and digits from `title` and `text` using regex.
2. **Tokenization & stopword removal**
   - Use `nltk.word_tokenize` and NLTK English stopwords.
   - Convert to lowercase and remove stopwords.
3. **Feature construction**
   - Apply **TF-IDF** separately to cleaned titles and texts:
     ```python
     TfidfVectorizer(max_features=1000)
     ```
   - Convert sparse matrices to arrays, then concatenate:
     ```python
     X = np.concatenate((title_tfidf, text_tfidf), axis=1)
     y = df['label']
     ```
4. **Train–test split**
   - 80/20 split with stratification on labels:
     ```python
     train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
     ```

### 3.2 Models

The notebook trains and evaluates:

- **Logistic Regression** (`sklearn.linear_model.LogisticRegression`)
- **KNN** (`sklearn.neighbors.KNeighborsClassifier`)
- **SVM (linear)** (`sklearn.svm.SVC(kernel='linear')`)
- **Gaussian Naive Bayes** (`sklearn.naive_bayes.GaussianNB`)
- **Decision Tree** (`sklearn.tree.DecisionTreeClassifier`)
- **Random Forest** (`sklearn.ensemble.RandomForestClassifier`)
- **XGBoost** (`xgboost.XGBClassifier`)

For each model you will see:

- Accuracy on the test set
- `classification_report` (precision, recall, F1 per class)
- Confusion matrix heatmap (via `seaborn.heatmap`)

### 3.3 SHAP Interpretability

For interpretability, the notebook focuses on **tree-based models**:

- **Random Forest**
- **XGBoost**

It uses `shap.TreeExplainer` to compute SHAP values and generates:

- **Global explanations**
  - Dot summary plot (feature importance and effect direction)
  - Bar summary plot (overall importance ranking)
- **Local explanations**
  - Waterfall plot for a single test instance (how each feature pushed the prediction toward real or fake)

These plots help you understand **which words/TF-IDF features** drive fake-news predictions.

---

## 4. Environment & Dependencies

### 4.1 Python Version

- Recommended: **Python 3.9+**

### 4.2 Required Packages

Install via `pip`:

```bash
pip install \
  numpy \
  pandas \
  scikit-learn \
  matplotlib \
  seaborn \
  nltk \
  xgboost \
  shap

