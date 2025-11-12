# import os, re, json, time, pickle
# from pathlib import Path
# from typing import List, Optional, Dict

# import numpy as np
# import joblib
# import streamlit as st
# import re 

# # -------------------- Page & Style --------------------
# st.set_page_config(page_title="coherenteyes — Fake News Detector", page_icon="🧿", layout="centered")
# st.markdown("""
# <style>
# .block-container {padding-top: 2rem; max-width: 880px;}
# label[data-baseweb="typography"] {font-weight:600;}
# .small {color:#9aa0a6; font-size:0.9rem;}
# .result-card {border:1px solid #2a2a2a; border-radius:14px; padding:16px; margin-top:12px;}
# /* Enhanced Badge Style for UX */
# .badge {display:inline-block; padding:8px 18px; border-radius:999px; font-weight:800; font-size:1.2rem; letter-spacing:1px; margin-bottom: 10px;}
# .badge-fake {background:#402424; color:#ff6b6b; border:2px solid #ff6b6b; animation: pulse-fake 1s infinite alternate;}
# .badge-real {background:#234737; color:#12d492; border:2px solid #12d492; animation: pulse-real 1s infinite alternate;}
# .stMetric {background-color: #1a1a1a; border-radius: 8px; padding: 10px 10px 10px 20px;}
# .stMetric > div:first-child {font-size: 0.9rem !important;} /* Label */
# .stMetric > div:nth-child(2) {font-size: 1.8rem !important; font-weight: bold;} /* Value */

# /* Keyframe Animations for Visual Appeal */
# @keyframes pulse-fake {
#     from {box-shadow: 0 0 0px #ff6b6b;}
#     to {box-shadow: 0 0 10px #ff6b6b;}
# }
# @keyframes pulse-real {
#     from {box-shadow: 0 0 0px #12d492;}
#     to {box-shadow: 0 0 10px #12d492;}
# }
# /* Removed custom style for example buttons */
# hr{border-color:#2a2a2a;}
# </style>
# """, unsafe_allow_html=True)

# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# # --- Supporting Functions (Path Resolver, Text Utils, Lazy TF, Loaders, Predictors) ---
# HERE = Path(__file__).resolve().parent
# CANDIDATE_DIRS = [HERE, HERE / "model", HERE.parent / "model"]

# def resolve_path(p: str) -> Optional[str]:
#     pth = Path(p)
#     if pth.is_absolute():
#         return str(pth) if pth.exists() else None
#     for base in CANDIDATE_DIRS:
#         cand = (base / pth).resolve()
#         if cand.exists():
#             return str(cand)
#     return None

# def any_exists(paths: List[str]) -> Optional[str]:
#     for raw in paths:
#         rp = resolve_path(raw)
#         if rp:
#             return rp
#     return None

# STOP = set("""
# a an the and or but if then else of to in for on at by from is are was were be been being as it its that this
# with about into through during before after above below up down out over under again further once here there
# when where why how all any both each few more most other some such no nor not only own same so than too very
# can will just don should now
# """.split())

# URL_REGEX = re.compile(r"http\S+|www\.\S+")
# MENTION_HASHTAG_REGEX = re.compile(r"[@#]\w+")
# PUNCT_ETC_REGEX = re.compile(r"[^a-z0-9\s']")

# def clean_min(s: str) -> str:
#     if not isinstance(s, str): return ""
#     s = s.strip().lower()
#     s = URL_REGEX.sub(" ", s)
#     s = MENTION_HASHTAG_REGEX.sub(" ", s)
#     s = PUNCT_ETC_REGEX.sub(" ", s)
#     toks = [t for t in s.split() if t not in STOP]
#     return " ".join(toks)

# def _tf():
#     import tensorflow as tf
#     return tf

# def _keras_pad():
#     from tensorflow.keras.preprocessing.sequence import pad_sequences
#     return pad_sequences

# def _keras_tok_from_json():
#     from tensorflow.keras.preprocessing.text import tokenizer_from_json
#     return tokenizer_from_json

# PATHS: Dict[str, Dict] = {
#     "LSTM (Keras)": {
#         "model": "fake_news_lstm.h5",
#         "tokenizer": ["tokenizer.joblib", "tokenizer.pkl", "tokenizer.json"],
#         "max_len": 256
#     },
#     "Logistic Regression (TF-IDF)": {
#         "model": "logistic_regression.joblib",
#         "vectorizer": "tfidf_vectorizer.joblib",
#     },
#     "SVM (TF-IDF)": {
#         "model": "svm.joblib",
#         "vectorizer": "tfidf_vectorizer.joblib",
#     },
#     "Naive Bayes (TF-IDF)": {
#         "model": "naive_bayes.joblib",
#         "vectorizer": "tfidf_vectorizer.joblib",
#     },
#     "Random Forest (TF-IDF)": {
#         "model": "random_forest.joblib",
#         "vectorizer": "tfidf_vectorizer.joblib",
#     },
# }

# def available_models() -> List[str]:
#     opts = []
#     for name, cfg in PATHS.items():
#         required_files = [cfg["model"]]
#         if "tokenizer" in cfg:
#             required_files.extend(cfg["tokenizer"])
#         if "vectorizer" in cfg:
#             required_files.append(cfg["vectorizer"])
            
#         is_available = True
#         for f in required_files:
#             if isinstance(f, list):
#                 if not any_exists(f):
#                     is_available = False
#                     break
#             else:
#                 if not resolve_path(f):
#                     is_available = False
#                     break
        
#         if is_available:
#             opts.append(name)
            
#     return opts or ["(no models found)"]

# @st.cache_resource(show_spinner=False)
# def load_lstm(model_file: str, tok_files: List[str], max_len: int):
#     tf = _tf()
#     model = tf.keras.models.load_model(model_file) 

#     tok = None
#     resolved_tok_files = [resolve_path(f) for f in tok_files if resolve_path(f)]
#     for p in resolved_tok_files:
#         if p and p.endswith((".joblib", ".pkl")):
#             try:
#                 obj = joblib.load(p) if p.endswith(".joblib") else pickle.load(open(p, "rb"))
#                 if hasattr(obj, "texts_to_sequences"):
#                     tok = obj
#                 else: 
#                     tok = _keras_tok_from_json()(obj)
#                 break
#             except Exception:
#                 pass
#         if p and p.endswith(".json"):
#             try:
#                 with open(p, "r", encoding="utf-8") as f:
#                     tok = _keras_tok_from_json()(json.load(f))
#                 break
#             except Exception:
#                 pass
    
#     if tok is None:
#         raise FileNotFoundError(f"Tokenizer not found for LSTM (looked for {tok_files}).")
#     return model, tok, max_len

# @st.cache_resource(show_spinner=False)
# def load_sklearn(model_file: str, vec_file: str):
#     clf = joblib.load(model_file)
#     vec = joblib.load(vec_file)
#     return clf, vec

# def predict_lstm(texts: List[str], model, tok, max_len: int) -> np.ndarray:
#     pad = _keras_pad()
#     seqs = tok.texts_to_sequences([clean_min(t) for t in texts])
#     X = pad(seqs, maxlen=max_len, padding="post", truncating="post")
#     p = model.predict(X, verbose=0).reshape(-1)
#     return p

# def predict_sklearn(texts: List[str], clf, vec) -> np.ndarray:
#     X = vec.transform([clean_min(t) for t in texts])
#     if hasattr(clf, "predict_proba"):
#         idx = int(np.where(clf.classes_ == 1)[0][0]) if 1 in clf.classes_ else -1
#         return clf.predict_proba(X)[:, idx]
#     if hasattr(clf, "decision_function"):
#         z = clf.decision_function(X)
#         return 1/(1+np.exp(-z))
#     return clf.predict(X).astype(float)


# # -------------------- UI (Cleaned Input Section) --------------------

# st.markdown("### 🧿 coherenteyes — Multi-Model Fake News Detector")
# st.caption("Select a model → paste a claim or article → **Classify**. Results show **REAL/FAKE** prediction and probability scores.")

# # --- Configuration Section (Model Selection) ---
# st.subheader("Configuration")
# col_model, col_thresh = st.columns([2, 1])

# with col_model:
#     choices = available_models()
#     selected = st.selectbox("Select Model Architecture", choices, index=0)

# with col_thresh:
#     threshold = st.slider("FAKE Decision Threshold", 0.05, 0.95, 0.50, 0.01)

# st.markdown("---")

# # --- Example & Input Section (Cleaned) ---
# st.subheader("Input Text")

# # Define English examples (used only for display reference)
# ex1 = "Drinking salt water every morning helps rejuvenate the body by 20 years, highly recommended by doctors." 
# ex2 = "NASA has successfully collected new rock samples from a massive volcanic crater on Mars, confirming the presence of ancient water flow." 
# ex3 = "Starting in 2030, all cash transactions in Vietnam will be mandatory for tracking via an embedded microchip." 

# # Display the example claims as reference text
# st.markdown("Use these claims as examples for manual testing:")
# st.markdown(f"""
# <div class="small" style="padding-left: 10px;">
# **1.** (Health Hoax): <i>{ex1}</i><br>
# **2.** (Factual News): <i>{ex2}</i><br>
# **3.** (Conspiracy Claim): <i>{ex3}</i>
# </div>
# """, unsafe_allow_html=True)
# st.markdown("---")


# # --- Input Text Area ---
# # Initialize session state for text area
# if "text" not in st.session_state:
#     st.session_state["text"] = ""
    
# # Set default value to empty string since buttons are removed
# text = st.text_area(
#     "Paste your Article or Claim Text here",
#     value="", 
#     height=200,
#     placeholder="Paste your news claim or article text here...",
#     key="input_text_area" 
# )

# classify_btn = st.button("Classify Claim", use_container_width=True, type="primary")

# # --- CLASSIFICATION LOGIC ---
# if classify_btn and text and selected != "(no models found)":
#     # 1. Prediction logic
#     st.markdown("---")
#     st.subheader("Classification Result ✨")

#     with st.spinner(f"Classifying with **{selected}**..."):
#         try:
#             # --- Load and predict logic ---
#             p_fake = 0.0

#             if selected.startswith("LSTM"):
#                 cfg = PATHS[selected]
#                 model_file = resolve_path(cfg["model"])
#                 tok_files = cfg["tokenizer"]
#                 max_len = cfg["max_len"]
                
#                 if not model_file:
#                     st.error(f"FATAL ERROR: LSTM model file not found at {cfg['model']}.")
#                     st.stop()
                    
#                 model, tok, _ = load_lstm(model_file, tok_files, max_len)
#                 p_fake = predict_lstm([text], model, tok, max_len)[0]
                
#             else:
#                 cfg = PATHS[selected]
#                 model_file = resolve_path(cfg["model"])
#                 vec_file = resolve_path(cfg["vectorizer"])
                
#                 if not model_file or not vec_file:
#                     st.error(f"FATAL ERROR: Scikit-learn model or vectorizer file missing for {selected}.")
#                     st.stop()
                    
#                 clf, vec = load_sklearn(model_file, vec_file)
#                 p_fake = predict_sklearn([text], clf, vec)[0]
            
#             # 2. Result calculation
#             prediction = "FAKE" if p_fake >= threshold else "REAL"
#             badge_class = "badge-fake" if prediction == "FAKE" else "badge-real"
#             p_real = 1.0 - p_fake

#             # 3. Display Results
#             col_pred, col_real, col_fake = st.columns([1.5, 1.2, 1.2])

#             with col_pred:
#                 st.markdown("##### Final Prediction")
#                 st.markdown(f'<div class="badge {badge_class}">{prediction}</div>', unsafe_allow_html=True)
                
#             with col_fake:
#                 st.metric("Probability (FAKE)", f"{p_fake*100:.2f}%", delta_color="inverse")
            
#             with col_real:
#                 st.metric("Probability (REAL)", f"{p_real*100:.2f}%", delta_color="normal")


#             st.markdown(f"""
#             <p class="small" style="margin-top: 20px;">
#             The model predicted **{prediction}** with a calculated P(fake) of {p_fake:.4f}. 
#             (Decision based on the set threshold of {threshold:.2f}.)
#             </p>
#             """, unsafe_allow_html=True)

#         except Exception as e:
#             st.error(f"CLASSIFICATION ERROR: An issue occurred during model prediction. Please check your model files.")
#             st.exception(e)

# elif classify_btn:
#     if not text:
#         st.warning("Please paste an article/claim into the text box to classify.")
#     elif selected == "(no models found)":
#         st.error("No compatible model files were found in the expected directories (./model or ../model). Please check your files.")

# app_gradio.py
import os, re, json, time, pickle
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import joblib
import gradio as gr
from functools import lru_cache

# -------------------- Global Config & Styles --------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

APP_TITLE = "🧿 coherenteyes — Multi-Model Fake News Detector (Gradio)"
APP_DESC = "Select a model → paste a claim or article → Classify. Results show REAL/FAKE and probability scores."
THEME_CSS = """
<style>
/* Container width */
.gradio-container {max-width: 890px !important; margin: 0 auto;}

/* Result card look */
.result-card {border:1px solid #2a2a2a; border-radius:14px; padding:16px; margin-top:12px;}
.small {color:#9aa0a6; font-size:0.9rem;}

/* Badges */
.badge {display:inline-block; padding:8px 18px; border-radius:999px; font-weight:800; font-size:1.1rem; letter-spacing:1px; margin-bottom:8px;}
.badge-fake {background:#402424; color:#ff6b6b; border:2px solid #ff6b6b; animation: pulse-fake 1s infinite alternate;}
.badge-real {background:#234737; color:#12d492; border:2px solid #12d492; animation: pulse-real 1s infinite alternate;}
@keyframes pulse-fake {from {box-shadow: 0 0 0px #ff6b6b;} to {box-shadow: 0 0 10px #ff6b6b;}}
@keyframes pulse-real {from {box-shadow: 0 0 0px #12d492;} to {box-shadow: 0 0 10px #12d492;}}
</style>
"""

# -------------------- Paths, Cleaning, Utilities --------------------
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

STOP = set("""
a an the and or but if then else of to in for on at by from is are was were be been being as it its that this
with about into through during before after above below up down out over under again further once here there
when where why how all any both each few more most other some such no nor not only own same so than too very
can will just don should now
""".split())

URL_REGEX = re.compile(r"http\S+|www\.\S+")
MENTION_HASHTAG_REGEX = re.compile(r"[@#]\w+")
PUNCT_ETC_REGEX = re.compile(r"[^a-z0-9\s']")

def clean_min(s: str) -> str:
    if not isinstance(s, str): 
        return ""
    s = s.strip().lower()
    s = URL_REGEX.sub(" ", s)
    s = MENTION_HASHTAG_REGEX.sub(" ", s)
    s = PUNCT_ETC_REGEX.sub(" ", s)
    toks = [t for t in s.split() if t not in STOP]
    return " ".join(toks)

def _tf():
    import tensorflow as tf
    return tf

def _keras_pad():
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    return pad_sequences

def _keras_tok_from_json():
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    return tokenizer_from_json

# -------------------- Model Registry --------------------
PATHS: Dict[str, Dict] = {
    "LSTM (Keras)": {
        "model": "fake_news_lstm.h5",
        "tokenizer": ["tokenizer.joblib", "tokenizer.pkl", "tokenizer.json"],
        "max_len": 256
    },
    "Logistic Regression (TF-IDF)": {
        "model": "logistic_regression.joblib",
        "vectorizer": "tfidf_vectorizer.joblib",
    },
    "SVM (TF-IDF)": {
        "model": "svm.joblib",
        "vectorizer": "tfidf_vectorizer.joblib",
    },
    "Naive Bayes (TF-IDF)": {
        "model": "naive_bayes.joblib",
        "vectorizer": "tfidf_vectorizer.joblib",
    },
    "Random Forest (TF-IDF)": {
        "model": "random_forest.joblib",
        "vectorizer": "tfidf_vectorizer.joblib",
    },
}

def available_models() -> List[str]:
    opts = []
    for name, cfg in PATHS.items():
        required_files = [cfg["model"]]
        if "tokenizer" in cfg:
            required_files.extend(cfg["tokenizer"])
        if "vectorizer" in cfg:
            required_files.append(cfg["vectorizer"])

        is_available = True
        for f in required_files:
            if isinstance(f, list):
                if not any_exists(f):
                    is_available = False
                    break
            else:
                if not resolve_path(f):
                    is_available = False
                    break
        if is_available:
            opts.append(name)
    return opts

# -------------------- Cached Loaders --------------------
@lru_cache(maxsize=8)
def load_lstm_cached(model_file: str, tok_files_key: str, max_len: int):
    """Cache by concrete resolved paths to avoid reloading each time."""
    tf = _tf()
    model = tf.keras.models.load_model(model_file)

    tok = None
    tok_files = json.loads(tok_files_key)
    resolved_tok_files = [resolve_path(f) for f in tok_files if resolve_path(f)]
    for p in resolved_tok_files:
        if p and p.endswith((".joblib", ".pkl")):
            try:
                obj = joblib.load(p) if p.endswith(".joblib") else pickle.load(open(p, "rb"))
                if hasattr(obj, "texts_to_sequences"):
                    tok = obj
                else:
                    tok = _keras_tok_from_json()(obj)
                break
            except Exception:
                pass
        if p and p.endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    tok = _keras_tok_from_json()(json.load(f))
                break
            except Exception:
                pass

    if tok is None:
        raise FileNotFoundError(f"Tokenizer not found (looked for {tok_files}).")
    return model, tok, max_len

@lru_cache(maxsize=16)
def load_sklearn_cached(model_file: str, vec_file: str):
    clf = joblib.load(model_file)
    vec = joblib.load(vec_file)
    return clf, vec

# -------------------- Predictors --------------------
def predict_lstm(texts: List[str], model, tok, max_len: int) -> np.ndarray:
    pad = _keras_pad()
    seqs = tok.texts_to_sequences([clean_min(t) for t in texts])
    X = pad(seqs, maxlen=max_len, padding="post", truncating="post")
    p = model.predict(X, verbose=0).reshape(-1)
    return p

def predict_sklearn(texts: List[str], clf, vec) -> np.ndarray:
    X = vec.transform([clean_min(t) for t in texts])
    # Prefer calibrated proba when available
    if hasattr(clf, "predict_proba"):
        # ensure class index of label 1
        try:
            idx = int(np.where(clf.classes_ == 1)[0][0])
        except Exception:
            idx = -1
        return clf.predict_proba(X)[:, idx]
    if hasattr(clf, "decision_function"):
        z = clf.decision_function(X)
        return 1 / (1 + np.exp(-z))
    # fallback: hard labels as float
    return clf.predict(X).astype(float)

# -------------------- Inference Wrapper --------------------
def classify(text: str, model_name: str, threshold: float) -> Tuple[str, str, str, str]:
    if not text.strip():
        return "", "", "", '<div class="small">Please paste a claim/article before running classification.</div>'

    try:
        p_fake = 0.0
        if model_name.startswith("LSTM"):
            cfg = PATHS[model_name]
            model_file = resolve_path(cfg["model"])
            tok_files = cfg["tokenizer"]
            max_len = cfg["max_len"]
            if not model_file:
                raise FileNotFoundError(f"LSTM model file not found at {cfg['model']}.")
            model, tok, _ = load_lstm_cached(model_file, json.dumps(tok_files), max_len)
            p_fake = float(predict_lstm([text], model, tok, max_len)[0])
        else:
            cfg = PATHS[model_name]
            model_file = resolve_path(cfg["model"])
            vec_file = resolve_path(cfg["vectorizer"])
            if not model_file or not vec_file:
                raise FileNotFoundError(f"Missing model/vectorizer for {model_name}.")
            clf, vec = load_sklearn_cached(model_file, vec_file)
            p_fake = float(predict_sklearn([text], clf, vec)[0])

        prediction = "FAKE" if p_fake >= threshold else "REAL"
        badge_class = "badge-fake" if prediction == "FAKE" else "badge-real"
        p_real = 1.0 - p_fake

        header_html = f'<div class="badge {badge_class}">{prediction}</div>'
        fake_txt = f"{p_fake*100:.2f}%"
        real_txt = f"{p_real*100:.2f}%"
        note = (
            f'<div class="small">P(fake)={p_fake:.4f} | '
            f"Threshold={threshold:.2f} ⇒ predicted <b>{prediction}</b>.</div>"
        )
        return header_html, real_txt, fake_txt, note

    except Exception as e:
        err = (
            '<div class="result-card"><b>CLASSIFICATION ERROR</b><br>'
            'An issue occurred during model prediction. '
            'Please verify your model files are present in <code>./model</code> or <code>../model</code>.<br>'
            f'<div class="small">Details: {type(e).__name__}: {str(e)}</div></div>'
        )
        return "", "", "", err

# -------------------- Model Scan (for UI refresh) --------------------
def scan_models() -> List[str]:
    opts = available_models()
    return opts if opts else ["(no models found)"]

# -------------------- Examples (Vietnam context, in English) --------------------
EXAMPLES = [
    # Health rumor in VN context
    "Drinking salt water every morning helps rejuvenate the body by 20 years, widely recommended by doctors in Ho Chi Minh City.",
    # Science news style
    "Vietnamese researchers at VNU-HCMC report a low-cost water filtration breakthrough tested across Mekong Delta communities.",
    # Conspiracy/policy rumor
    "Starting in 2030, all cash transactions in Vietnam will be tracked by an embedded microchip in banknotes.",
]

# -------------------- Build Gradio UI --------------------
def build_ui():
    with gr.Blocks(fill_width=True, head=THEME_CSS) as demo:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESC)

        with gr.Row():
            model_dd = gr.Dropdown(
                label="Select Model Architecture",
                choices=scan_models(),
                value=None,
                allow_custom_value=False,
                info="Models are discovered in ./model or ../model.",
                scale=2
            )
            threshold = gr.Slider(
                label="FAKE Decision Threshold",
                minimum=0.05, maximum=0.95, value=0.50, step=0.01,
                scale=1
            )
            rescan_btn = gr.Button("🔄 Rescan models", variant="secondary")

        gr.Markdown("---")
        gr.Markdown("### Input Text")
        gr.Markdown(
            "<div class='small'>Use these Vietnam-context examples for quick testing (in English):</div>",
            elem_id="examples-caption"
        )

        # Examples populate the textbox when clicked
        examples_comp = gr.Examples(
            examples=[[ex] for ex in EXAMPLES],
            inputs=[gr.Textbox(visible=False)],  # placeholder to satisfy Gradio; we will set manually
            label=None
        )

        text = gr.Textbox(
            label="Paste your Article or Claim Text here",
            placeholder="Paste your news claim or article text here...",
            lines=10
        )
        # Wire examples -> textbox manually (Gradio's Examples ties to inputs; we emulate for a single box)
        def set_example(i: int):
            return EXAMPLES[i] if 0 <= i < len(EXAMPLES) else ""
        for i, _ in enumerate(EXAMPLES):
            # Create small buttons beneath examples to load into textbox
            pass

        with gr.Row():
            classify_btn = gr.Button("Classify Claim", variant="primary")
            clear_btn = gr.Button("Clear")

        gr.Markdown("---")
        gr.Markdown("### Classification Result ✨")

        with gr.Row():
            pred_badge = gr.HTML()
            real_prob = gr.Textbox(label="Probability (REAL)", interactive=False)
            fake_prob = gr.Textbox(label="Probability (FAKE)", interactive=False)

        note_md = gr.HTML()

        # ------ Callbacks ------
        def do_rescan():
            choices = scan_models()
            # Reset selected value if no models found
            selected = None if "(no models found)" in choices and len(choices) == 1 else (choices[0] if choices else None)
            return gr.update(choices=choices, value=selected)

        rescan_btn.click(fn=do_rescan, outputs=[model_dd])

        classify_btn.click(
            fn=classify,
            inputs=[text, model_dd, threshold],
            outputs=[pred_badge, real_prob, fake_prob, note_md]
        )

        clear_btn.click(lambda: ("", "", "", ""), None, [pred_badge, real_prob, fake_prob, note_md])
        clear_btn.click(lambda: "", None, [text])

        # Make example tiles clickable to fill the textbox
        # (Gradio Examples isn't bound to 'text' by default in this setup, so we add buttons.)
        with gr.Row():
            ex_btns = []
            for i, ex in enumerate(EXAMPLES, start=1):
                b = gr.Button(f"Use Example {i}", size="sm")
                b.click(lambda idx=i-1: set_example(idx), outputs=[text])
                ex_btns.append(b)

    return demo

if __name__ == "__main__":
    demo = build_ui()
    # You can change server_name / server_port as needed
    demo.launch()
