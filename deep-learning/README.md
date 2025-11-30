Fake News Detection on WELFake (Deep Learning + SHAP for CNN–LSTM & CNN–PCA)

This directory contains the deep-learning pipeline for detecting fake news on the WELFake dataset using neural architectures. The notebooks (welfake-glove.ipynb, welfake-glove-with-shap.ipynb, and welfake_bert.ipynb) walk through data preprocessing, text embedding, training multi-architecture deep-learning models, and performing SHAP interpretability for selected models.

Project Overview

The notebooks implement the following workflow:

Load and process text from the WELFake dataset

Clean and prepare text for deep-learning models (tokenization, padding, sequence handling)

Train and evaluate multiple deep architectures:

CNN–LSTM (GloVe embeddings)

CNN with PCA-reduced GloVe embeddings

BERT-base fine-tuning for document classification

Integrate explainability using:

SHAP for CNN–LSTM (DeepExplainer / GradientExplainer depending on GPU availability)

SHAP for CNN–PCA (SamplingExplainer for lightweight explanations)

Visualize:

Global feature effects (summary plots)

Local token-level SHAP attributions for individual predictions

Note:
SHAP is not applied to BERT due to extremely high computational cost and memory requirements. Transformer-based SHAP typically requires:

multi-GPU or TPU resources,

expensive sampling-based estimators,

model-partitioning techniques not feasible in constrained environments.

Thus, SHAP explanations are only provided for CNN–LSTM and CNN–PCA in this version.

Model Summary
CNN–LSTM (GloVe)

Word-level embeddings (100d GloVe)

1D Convolution for local lexical pattern detection

LSTM layer for long-sequence contextual modelling

Dense layer for binary fake/real prediction

CNN–PCA (GloVe)

PCA-compressed embeddings for lower compute cost

Lightweight CNN classifier

Faster inference and significantly faster SHAP evaluation

BERT-base

Transformer encoder with self-attention

Fine-tuned on WELFake titles + text

Highest accuracy among models, but not SHAP-interpretable in this version

Interpretability (SHAP)

SHAP is used to provide model-agnostic and model-specific explanations:

CNN–LSTM → DeepExplainer or GradientExplainer

CNN–PCA → SamplingExplainer (low-cost fallback)

Interpretability outputs include:

SHAP summary plots showing globally influential tokens

Force plots for local explanations of individual articles

Analysis of whether models rely on meaningful linguistic patterns or dataset artifacts

Running the Models

Ensure the dataset is placed in the same directory or adjust the path:

df = pd.read_csv("WELFake_Dataset.csv")


Run the notebooks:

welfake-glove.ipynb → CNN–LSTM

welfake-glove-with-shap.ipynb → CNN–LSTM + SHAP

welfake_bert.ipynb → BERT fine-tuning
