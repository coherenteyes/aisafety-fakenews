# Fake News Detection on WELFake (Deep Learning + SHAP for CNN–LSTM & CNN–PCA)

This directory contains the **deep-learning pipeline** for detecting fake news on the **WELFake** dataset using neural network architectures. The notebooks (`welfake-glove.ipynb`, `welfake-glove-with-shap.ipynb`, and `welfake_bert.ipynb`) walk through data preprocessing, embedding construction, model training, evaluation, and interpretable analysis using **SHAP** for selected models.

---

## Project Overview

The deep-learning workflow includes:

- Loading and preprocessing text from the **WELFake** dataset  
- Cleaning, tokenizing, padding, and preparing sequences for neural models  
- Training multiple architectures:
  - **CNN–LSTM (GloVe embeddings)**
  - **CNN with PCA-reduced GloVe embeddings**
  - **BERT-base** (transformer fine-tuning)
- Providing **model interpretability** via:
  - **SHAP for CNN–LSTM** (DeepExplainer / GradientExplainer depending on hardware)
  - **SHAP for CNN–PCA** (SamplingExplainer for faster estimations)
- Generating:
  - Global attribution plots (SHAP summary)
  - Local token-level explanations for individual predictions  

**Note:**  
SHAP is **not applied to BERT** due to high computational cost and memory requirements.  
Transformers require expensive model-partitioning or sampling techniques, which are not feasible in limited-resource environments.

---

## Model Summary

### **1. CNN–LSTM (GloVe)**
- Uses pretrained **GloVe embeddings**
- Extracts local n-gram patterns with **1D CNN**
- Captures long-range dependencies using **LSTM**
- Good balance of speed, accuracy, and interpretability
- Compatible with SHAP for token-level explanations

### **2. CNN–PCA (GloVe)**
- Applies **PCA** to compress 100d GloVe into lower dimensions  
- Lightweight model with reduced compute requirements  
- Extremely fast SHAP evaluation  
- Suitable for resource-constrained environments

### **3. BERT-base**
- Transformer encoder with self-attention  
- Fine-tuned on WELFake titles and body text  
- Achieves highest predictive performance  
- **Interpretability not included** due to SHAP constraints  

---

## Interpretability with SHAP

SHAP is used to interpret CNN–LSTM and CNN–PCA models:

- **DeepExplainer** or **GradientExplainer** for CNN–LSTM  
- **SamplingExplainer** for PCA-based models  
- Outputs include:
  - **Summary plots** identifying globally important words  
  - **Force plots** showing how individual tokens push predictions toward FAKE or REAL  
  - Insights into whether models rely on meaningful linguistic cues or dataset artifacts  

---

## Running the Models

Place the dataset in the same directory or adjust the file path:

```python
df = pd.read_csv("WELFake_Dataset.csv")
