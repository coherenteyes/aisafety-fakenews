
# 🧿 CoherentEyes — Multi-Model Fake News Detector

> Detect misinformation instantly using multiple machine learning models — from classical ML to transformer-based architectures — through a simple Gradio web interface.

---

## 🌐 Overview

**CoherentEyes** is an interactive web application designed for exploring how different AI models detect **fake news** and misinformation.  
Users can select among multiple models, input a news claim or short article, and instantly view predictions with probability scores.

This project combines **machine learning transparency**, **comparative explainability**, and **Vietnam-specific examples** to showcase the evolving landscape of misinformation detection.

---

## 🧠 Supported Models

The app supports multiple trained models stored under `/models` or loaded from Google Drive:

| Model Name | Type | Description |
|-------------|------|-------------|
| **TF-IDF + LogisticRegression** | Classical ML | Lightweight baseline using bag-of-words representation. |
| **LSTM (Bi-LSTM / GRU)** | Deep Learning | Sequential text classifier trained on `fake.csv` + `true.csv`. |
| **RAG / DAPT Variant** *(optional)* | Retrieval-Augmented | Extended experimental variant for factual grounding. |

Each model outputs:
- **Label:** `FAKE` or `REAL`
- **P(fake):** predicted probability
- **Confidence:** threshold-based classification indicator
- **Preprocessed Text:** version actually fed into the model

---

## 🧩 Features

- 🔘 **Model Selection Dropdown** — choose your model interactively  
- 🧾 **Input Claim / Article Box** — paste any short claim or paragraph  
- 🎛️ **Decision Threshold Slider** — adjust sensitivity of classification (`p(fake) ≥ τ`)  
- 🧪 **Example Buttons** — test preset claims from Vietnamese or global context  
- 🧮 **Probability + Cleaned Text Display** — see model reasoning trace  
- 🖤 **Dark Minimal UI** (custom CSS) with responsive layout  

Example usage:
> “Vietnam’s National Assembly passes a cybersecurity data-protection amendment for 2025.”

---

## 📸 Screenshot
<img width="1081" height="810" alt="Screenshot 2025-11-13 at 2 07 45 AM" src="https://github.com/user-attachments/assets/4f33ab95-2d2d-4087-8b41-7434bc54111c" />



