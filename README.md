# Project Summary — Fake News Detection Using LSTM and Traditional Machine Learning

## Objective
The goal of this project is to develop and evaluate models that can accurately classify news articles as **real** or **fake**, using both traditional machine learning and deep learning techniques.

---

## Dataset
The dataset includes two CSV files:
- **True.csv** — authentic, factual news articles  
- **Fake.csv** — fabricated or misleading news articles  

After merging and preprocessing:
- Each entry contains a combined **content** field (title + article text)  
- A **label** (`1 = real`, `0 = fake`) marks the ground truth  

---

## Data Preprocessing
- Removed punctuation, stopwords, and special characters  
- Tokenized and lemmatized text using **NLTK**  
- Combined title and body to form a richer text representation  
- Converted text into:
  - **TF-IDF vectors** for traditional models  
  - **Tokenized sequences** for the LSTM model (with padding and integer encoding)  

---

## Models and Methods

### Traditional Machine Learning
Implemented using **Scikit-learn**:
- Logistic Regression  
- Naive Bayes  
- Random Forest  
- Linear SVM  

These models used **TF-IDF vectorized features** for classification.

### Deep Learning (LSTM)
A **Sequential LSTM model** was built using **TensorFlow/Keras**, consisting of:
- Embedding layer  
- LSTM layer  
- Dense layers with ReLU and sigmoid activations  
- Dropout for regularization  

---

## Evaluation Metrics
All models were evaluated using:
- Accuracy  
- Precision, Recall, and F1-score  
- Confusion matrix  
- ROC and AUC analysis  

---

## Results

| Model | Description | Accuracy |
|:------|:-------------|:----------:|
| Logistic Regression | TF-IDF baseline | **0.9893** |
| Naive Bayes | Probabilistic classifier | **0.9404** |
| Random Forest | Ensemble model | **0.9978** |
| Linear SVM | Margin-based classifier | **0.9951** |
| **LSTM (Deep Learning)** | Sequence-based model | **0.9970** |

The **LSTM model** achieved outstanding performance, with validation accuracy improving steadily across epochs:

> **Epoch 20/20 → Accuracy: 0.9996 (train), 0.9970 (validation)**

This demonstrates that the LSTM model effectively captured semantic patterns in text, significantly outperforming traditional ML approaches.

---

## Conclusion
This project shows that while **traditional models with TF-IDF features** perform very well, **deep learning models** — especially **LSTM networks** — can achieve **near-perfect accuracy** by learning contextual and sequential dependencies in language.
