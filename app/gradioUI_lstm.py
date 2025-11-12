
# coherenteyes — Gradio app (LSTM-only)
# Run: python app_gradio.py

import os, re, json, pickle
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import joblib
import gradio as gr

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# -------- robust path resolver (./model or ../model) --------
HERE = Path(__file__).resolve().parent
CANDIDATE_DIRS = [HERE, HERE / "model", HERE.parent / "model"]

def resolve_path(p: str) -> Optional[str]:
    pth = Path(p)
    if pth.is_absolute():
        return str(pth) if pth.exists() else None
    for base in CANDIDATE_DIRS:
        cand = (base / pth).resolve()
        if cand.exists():
            return str(cand)
    return None

def any_exists(paths: List[str]) -> Optional[str]:
    for raw in paths:
        rp = resolve_path(raw)
        if rp:
            return rp
    return None

# -------- text cleaning (no nltk) --------
STOP = set("""
a an the and or but if then else of to in for on at by from is are was were be been being as it its that this
with about into through during before after above below up down out over under again further once here there
when where why how all any both each few more most other some such no nor not only own same so than too very
can will just don should now
""".split())

def clean_min(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip().lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[@#]\w+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    toks = [t for t in s.split() if t not in STOP]
    return " ".join(toks)

# -------- lazy TF + loaders --------
def _tf():
    import tensorflow as tf
    return tf

def _pad_sequences():
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences

def _tokenizer_from_json():
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    return tokenizer_from_json

def load_lstm_artifacts() -> Tuple[object, object, int]:
    """
    Returns: (model, tokenizer, max_len)
    Looks for:
      - fake_news_lstm.h5
      - tokenizer.joblib | tokenizer.pkl | tokenizer.json
    """
    model_path = resolve_path("fake_news_lstm.h5")
    tok_path = any_exists(["tokenizer.joblib", "tokenizer.pkl", "tokenizer.json"])
    if not model_path or not tok_path:
        raise FileNotFoundError(
            "Could not locate LSTM artifacts.\n"
            f"Searched in: {', '.join(str(p) for p in CANDIDATE_DIRS)}\n"
            "Expected files: fake_news_lstm.h5 and tokenizer.(joblib|pkl|json)"
        )

    tf = _tf()
    model = tf.keras.models.load_model(model_path)

    tokenizer = None
    if tok_path.endswith(".joblib"):
        obj = joblib.load(tok_path)
        tokenizer = obj if hasattr(obj, "texts_to_sequences") else _tokenizer_from_json()(obj)
    elif tok_path.endswith(".pkl"):
        with open(tok_path, "rb") as f:
            obj = pickle.load(f)
        tokenizer = obj if hasattr(obj, "texts_to_sequences") else _tokenizer_from_json()(obj)
    elif tok_path.endswith(".json"):
        with open(tok_path, "r", encoding="utf-8") as f:
            tokenizer = _tokenizer_from_json()(json.load(f))

    if tokenizer is None:
        raise RuntimeError("Tokenizer could not be loaded.")

    max_len = 256  # adjust if your training used a different length
    return model, tokenizer, max_len

# -------- prediction --------
def predict(text: str, threshold: float, state: dict):
    if not text or not text.strip():
        return "—", 0.0, "", "Please paste a claim or short article."

    model = state.get("model")
    tokenizer = state.get("tokenizer")
    max_len = state.get("max_len", 256)

    if model is None or tokenizer is None:
        try:
            model, tokenizer, max_len = load_lstm_artifacts()
            state["model"], state["tokenizer"], state["max_len"] = model, tokenizer, max_len
        except Exception as e:
            return "—", 0.0, "", f"Model load error: {e}"

    clean = clean_min(text)
    pad_sequences = _pad_sequences()
    seqs = tokenizer.texts_to_sequences([clean])
    X = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

    tf = _tf()
    p = tf.convert_to_tensor(model.predict(X, verbose=0)).numpy().reshape(-1)
    # If your model outputs 2-unit softmax, switch to: p = model.predict(X)[:, 1]
    p_fake = float(p[0]) if p.shape == (1,) else float(p[-1])

    label = "FAKE" if p_fake >= float(threshold) else "REAL"
    return label, round(p_fake, 3), clean, ""

# -------- app (Gradio Blocks) --------
with gr.Blocks(title="coherenteyes — Fake News Detector") as demo:
    gr.Markdown(
        "<h2 style='margin-bottom:0'>🧿 coherenteyes</h2>"
        "<p style='color:#9aa0a6;margin-top:4px'>Paste a claim/article → Classify with your LSTM model (REAL / FAKE + probability).</p>"
    )

    state = gr.State({"model": None, "tokenizer": None, "max_len": 256})

    with gr.Row():
        threshold = gr.Slider(0.05, 0.95, value=0.50, step=0.01, label="Decision threshold (predict FAKE if P(fake) ≥ …)")

    # Examples (Vietnam context, English)
    ex1 = "Drinking salt water each morning makes you look 20 years younger, Vietnamese doctors officially advise."
    ex2 = "Vietnam’s National Assembly passes a cybersecurity data-protection amendment for 2025."
    ex3 = "From 2030, all cash transactions in Vietnam will be tracked by government-issued microchips."

    gr.Markdown(
        f"**Examples:** `{ex1}` · `{ex2}` · `{ex3}`"
    )

    with gr.Row():
        b1 = gr.Button("Use example 1")
        b2 = gr.Button("Use example 2")
        b3 = gr.Button("Use example 3")

    text = gr.Textbox(label="Paste claim or article", lines=6, placeholder="Type or paste here…")

    # Buttons fill the textbox
    b1.click(fn=lambda: ex1, outputs=text)
    b2.click(fn=lambda: ex2, outputs=text)
    b3.click(fn=lambda: ex3, outputs=text)

    classify = gr.Button("Classify", variant="primary")

    with gr.Row():
        label = gr.Label(label="Predicted label")
        prob = gr.Number(label="P(fake)", precision=3)

    cleaned = gr.Textbox(label="Cleaned text used by the model", lines=4, interactive=False)
    error = gr.Markdown()

    classify.click(
        fn=predict,
        inputs=[text, threshold, state],
        outputs=[label, prob, cleaned, error]
    )

if __name__ == "__main__":
    # Note: set server_name="0.0.0.0" if running in a container/remote
    demo.launch(server_name="0.0.0.0")
