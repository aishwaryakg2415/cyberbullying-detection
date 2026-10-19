"""SACENet demo — predictions come only from the trained checkpoints on disk.

Run:  streamlit run app/streamlit_app.py
Privacy: submitted text is processed in memory and never written to disk or logged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from html import escape as html_escape  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

CB_MODEL = Path(os.environ.get("SACENET_CB_MODEL", ROOT / "results/runs/mc7__muril__base__s13/checkpoint/model"))
SARC_MODEL = Path(os.environ.get("SACENET_SARC_MODEL", ROOT / "results/runs/sarcasm__muril__base__s13/checkpoint/model"))

st.set_page_config(page_title="SACENet demo", layout="centered")


@st.cache_resource(show_spinner="Loading trained models…")
def load(path: str):
    from src.inference.predictor import Predictor
    return Predictor(path, device=os.environ.get("SACENET_DEVICE", "auto"), calibrated=True)


def model_or_error(path: Path, what: str):
    if not path.exists():
        st.error(f"The trained {what} checkpoint was not found at `{path}`. "
                 "Train it with `python scripts/run_experiments.py --group mc7_backbones --seeds 13` "
                 "(or `sarcasm_component`). No fallback or simulated predictions are shown.")
        return None
    try:
        return load(str(path))
    except Exception as e:  # pragma: no cover - UI path
        st.error(f"Could not load the {what} model: {e}")
        return None


st.title("SACENet — Hinglish cyberbullying-type classifier")
st.caption("Research demo of a MuRIL model fine-tuned on MC-Hinglish1.0 (7 types). Outputs are model estimates, "
           "not judgements about people. Read the limitations below before use.")

tab1, tab2, tab3 = st.tabs(["Single comment", "Conversation (prototype)", "About & limitations"])

with tab1:
    lang = st.selectbox("Input language", ["Hinglish (Roman script) — supported",
                                           "English — out of domain, unreliable",
                                           "Hindi (Devanagari) — out of domain, unreliable"])
    text = st.text_area("Comment", height=100, placeholder="Type or paste a comment…")
    c1, c2 = st.columns(2)
    explain = c1.checkbox("Explain with SHAP (slower, ~10-60 s on CPU)")
    show_sarc = c2.checkbox("Show separate sarcasm-model score", value=True)
    if st.button("Analyse", type="primary") and text.strip():
        cb = model_or_error(CB_MODEL, "cyberbullying")
        if cb:
            with st.spinner("Predicting…"):
                p = cb.predict_proba([text])[0]
            labels = cb.labels
            p_cb = 1 - p[labels.index("not_cyberbullying")]
            top = labels[int(p.argmax())]
            st.metric("Most likely type", top.replace("_", " "), f"{p.max():.0%} (calibrated)")
            st.metric("P(any cyberbullying type)", f"{p_cb:.0%}")
            df = pd.DataFrame({"type": labels, "probability": p}).sort_values("probability", ascending=False)
            st.bar_chart(df.set_index("type"))
            if not lang.startswith("Hinglish"):
                st.warning("The model was trained only on Roman-script Hinglish; this input is out of domain.")
            if show_sarc:
                sm = model_or_error(SARC_MODEL, "sarcasm")
                if sm:
                    ps = sm.predict_proba([text])[0, 1]
                    st.info(f"Sarcasm model (trained on Swami et al. 2018 tweets): P(sarcastic) = {ps:.0%}. "
                            "This score is shown for information only; it does not change the prediction above.")
            if explain:
                from src.explainability.explain import shap_explain, to_words
                from src.data.text import clean_text
                with st.spinner("Computing SHAP attributions…"):
                    e = shap_explain(cb, [clean_text(text)], max_evals=200)[0]
                words, vals = to_words(e["tokens"], e["values"])
                vmax = max(1e-9, float(np.abs(vals).max()))
                # User text is HTML-escaped before being rendered with unsafe_allow_html.
                html = " ".join(
                    f"<span style='background: rgba({'227,73,72' if v > 0 else '42,120,214'},{min(1, abs(v) / vmax):.2f});"
                    f"padding:2px 4px;border-radius:4px;margin:1px;display:inline-block'>{html_escape(w)}</span>"
                    for w, v in zip(words, vals))
                st.markdown(f"Explained class: **{e['target_label']}**. Red pushes toward it, blue away.",
                            unsafe_allow_html=False)
                st.markdown(html, unsafe_allow_html=True)
                st.caption("Attributions show which words the model's output is sensitive to. They do not establish "
                           "intent or prove that a word caused harm.")

with tab2:
    st.warning("Prototype. The causal early-warning procedure was evaluated only on English in-game chat (CONDA) as a "
               "proxy. It has **not** been validated for Hinglish cyberbullying conversations.")
    conv = st.text_area("One message per line, in order", height=160)
    tau = st.slider("Alarm threshold on running max of P(cyberbullying)", 0.05, 0.95, 0.5, 0.05)
    if st.button("Analyse conversation") and conv.strip():
        cb = model_or_error(CB_MODEL, "cyberbullying")
        if cb:
            msgs = [m for m in conv.splitlines() if m.strip()]
            P = cb.predict_proba(msgs)
            p_cb = 1 - P[:, cb.labels.index("not_cyberbullying")]
            run = np.maximum.accumulate(p_cb)
            alarm = next((i for i, a in enumerate(run) if a >= tau), None)
            st.line_chart(pd.DataFrame({"P(CB) per message": p_cb, "running max": run},
                                       index=range(1, len(msgs) + 1)))
            st.write(f"Alarm at message **{alarm + 1}**." if alarm is not None else "No alarm raised.")
            st.caption("Each score uses only that message (no future messages).")

with tab3:
    st.markdown("""
**What this is.** A MuRIL encoder fine-tuned on MC-Hinglish1.0 (Rahman-Laskar et al., 2024), predicting one of seven
labels: not cyberbullying, age, gender, religion, mockery, abusive, offensive. Probabilities are temperature-scaled on
the validation split. See `MODEL_CARD.md` and `results/FINAL_EXPERIMENT_REPORT.md` for the measured test performance.

**Limitations.**
- Trained on ~6.7k Roman-script Twitter comments; performance on other platforms, scripts and topics is unknown.
- 18% of the official test items have near-duplicates in training data; scores on new data will likely be lower.
- `mockery` is the least reliable class. Sarcasm and indirect harassment remain hard.
- Labels mix target-based (age/gender/religion) and language-based (abusive/offensive) categories.
- The model can reflect annotation biases about religion, gender and dialect.

**Responsible use.** Use only to assist trained human moderators or for research. Do not use for automated
sanctions, profiling of individuals, or any decision about a person without human review.

**Privacy.** Text you enter is processed in memory and is not stored or logged by this app.
""")
