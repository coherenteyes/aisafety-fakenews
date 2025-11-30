<div align="center">

# 🧿 **coherenteyes**
### *AI Safety–Aligned Fake News Detection System*

A unified machine learning + deep learning pipeline for detecting fake news on the **WELFake** dataset,  
designed with a strong emphasis on **explainability**, **transparency**, and **AI Safety**.

</div>

---

# 📘 About This Project

**coherenteyes** is an AI Safety–oriented misinformation detection system that integrates  
**classical machine learning**, **deep learning architectures**, and **SHAP explainability**  
to provide transparent and interpretable predictions.

Beyond achieving high accuracy, the project aims to answer critical safety questions:

- *How do different models reason about fake vs. real news?*  
- *Do models rely on meaningful patterns or dataset artifacts?*  
- *How interpretable are decisions from linear models vs. neural networks?*  
- *How robust are these models under distribution shift or adversarial variations?*  

The project demonstrates how explainable AI can be used to **audit model behavior**,  
**identify failure modes**, and **support responsible deployment** in sensitive information ecosystems.

---

# 📁 Repository Structure

aisafety-fakenews/
│
├── data/
│ └── WELFake_Dataset.csv (not included)
│
├── machine-learning/
│ ├── README.md
│ └── welfake_ml.ipynb
│
├── deep-learning/
│ ├── README.md
│ ├── welfake-glove.ipynb
│ ├── welfake-glove-with-shap.ipynb
│ └── welfake_bert.ipynb
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
│
└── README.md (this file)


---

# 🚀 Project Overview

The repository contains **two complete modeling pipelines**:

---

## 🔷 1. Classical Machine Learning (TF–IDF Models)

Located in: `machine-learning/`

Includes:

- Text preprocessing + TF–IDF vectorization  
- Models:
  - Logistic Regression  
  - SVM  
  - KNN  
  - Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- **Interpretability:**  
  - SHAP for Logistic Regression  
  - SHAP for XGBoost (TreeSHAP)

Outputs:

- SHAP summary plots  
- Local force plots  
- Global token importance  
- Classical ML model comparison  

---

## 🔷 2. Deep Learning Models (CNN, LSTM, BERT)

Located in: `deep-learning/`

Includes:

- **CNN–LSTM (GloVe embeddings)**  
- **CNN–PCA (compressed embeddings)**  
- **BERT-base transformer**  
- **SHAP** for:
  - CNN–LSTM  
  - CNN–PCA  

Outputs:

- Token-level SHAP values  
- Interpretability for neural networks  
- BERT attention-based insights  
- Full training + evaluation pipeline  

---

# 📊 Model Performance Leaderboard



| Model | Type | Accuracy | Precision | Recall | F1-score | SHAP Support |
|-------|------|----------|-----------|--------|----------|--------------|
| **CNN–LSTM (GloVe)** | Deep Learning | **0.9821** | N/A | N/A | **0.9821** (Val Acc) | ✅ Yes |
| **BERT + CNN** | Transformer Hybrid | 0.9815 | N/A | N/A | 0.9815 | ❌ Too costly |
| **BERT + BiLSTM** | Transformer Hybrid | 0.9813 | N/A | N/A | 0.9813 | ❌ Too costly |
| **CNN–PCA (GloVe)** | Deep Learning | 0.9805 | N/A | N/A | 0.9805 | ✅ Yes |
| **CNN (GloVe)** | Deep Learning | 0.9714 | N/A | N/A | 0.9714 | ✅ Yes |
| **Random Forest** | Classical ML | **0.9660** | 0.966 | 0.966 | **0.966** | ❌ No |
| **Linear SVM** | Classical ML | 0.9650 | 0.965 | 0.965 | 0.965 | ❌ No |
| **Logistic Regression** | Classical ML | 0.9630 | 0.963 | 0.963 | 0.963 | ✅ Yes |
| **AdaBoost** | Classical ML | 0.9520 | 0.952 | 0.952 | 0.952 | ❌ No |
| **XGBoost** | Classical ML | 0.9410 | 0.941 | 0.941 | 0.941 | ✅ Yes |
| **Decision Tree** | Classical ML | 0.8980 | 0.899 | 0.899 | 0.899 | ❌ No |
| **Gaussian Naive Bayes** | Classical ML | 0.8570 | 0.857 | 0.857 | 0.857 | ❌ No |
| **KNN (k=3)** | Classical ML | 0.7740 | 0.770 | 0.770 | 0.770 | ❌ No |


---

# 🧠 AI Safety Design Principles

coherenteyes focuses on the following safety principles:

### ✔️ Transparency  
- SHAP explanations  
- Token-level interpretability  
- Attention visualization  

### ✔️ Robustness  
Evaluated across:
- Different architectures  
- Long vs. short text  
- Simple vs. complex features  

### ✔️ Failure Mode Analysis  
Identifies issues such as:
- Dataset artifacts  
- Keyword reliance  
- Overconfidence  

### ✔️ Responsible Deployment  
Warns about the risks of:
- False positives (unwanted censorship)  
- False negatives (misinformation spread)  
- Distribution shift failures  

---

# ⚙️ Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```
If you use this repository in your research, cite:
coherenteyes (2025). AI Safety–Aligned Fake News Detection.

🤝 Contributing

Contributions are welcome!
Please ensure additions maintain:

Clarity

Reproducibility

Transparency

Safety alignment

