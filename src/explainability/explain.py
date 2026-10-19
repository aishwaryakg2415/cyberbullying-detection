"""Token attributions (SHAP Partition explainer) + checks of what they are worth.

Attributions describe how the *model's output* changes when parts of the input
are masked. They are not evidence of a speaker's intent, and a high attribution
does not show that a word "caused" harm. We therefore report two checks:

* faithfulness (deletion test, cf. ERASER "comprehensiveness", DeYoung et al. 2020):
  drop in P(predicted class) after deleting the top-k attributed words vs. k
  random words;
* stability: agreement of SHAP rankings across evaluation budgets, and between
  SHAP and leave-one-word-out occlusion.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.stats import spearmanr


def shap_explain(predictor, texts: list[str], target: list[int] | None = None, max_evals: int = 500,
                 batch_size: int = 32) -> list[dict]:
    """Subword-level SHAP values for the target class of each text (default: predicted class)."""
    import shap

    def f(xs):
        return predictor.predict_proba([str(x) for x in xs], batch_size=batch_size, clean=False)

    masker = shap.maskers.Text(predictor.tokenizer)
    explainer = shap.Explainer(f, masker, output_names=predictor.labels, algorithm="partition")
    out = []
    for i, t in enumerate(texts):
        p = f([t])[0]
        k = int(p.argmax()) if target is None else int(target[i])
        t0 = time.perf_counter()
        sv = explainer([t], max_evals=max_evals, batch_size=batch_size, silent=True)
        secs = time.perf_counter() - t0
        vals = np.asarray(sv.values[0])[:, k]
        toks = [str(x) for x in sv.data[0]]
        out.append({"text": t, "target": k, "target_label": predictor.labels[k], "prob": p.tolist(),
                    "tokens": toks, "values": vals.tolist(),
                    "base_value": float(np.asarray(sv.base_values[0])[k]), "seconds": secs,
                    "max_evals": max_evals})
    return out


def to_words(tokens: list[str], values: list[float]) -> tuple[list[str], np.ndarray]:
    """Aggregate subword attributions to whitespace words (sum), using SHAP's token strings.

    SHAP's Text masker attaches the whitespace that FOLLOWS a token to it
    (e.g. ['', 'ye', 'h ', 'banda ', ...] for MuRIL), so a token ending in
    whitespace closes the current word. Empty special-token slots are skipped.
    """
    words, wv = [], []
    cur, cur_v = "", 0.0
    for tok, v in zip(tokens, values):
        if tok == "":
            continue
        cur += tok.strip()
        cur_v += float(v)
        if tok != tok.rstrip() and cur:
            words.append(cur)
            wv.append(cur_v)
            cur, cur_v = "", 0.0
    if cur:
        words.append(cur)
        wv.append(cur_v)
    return words, np.array(wv, dtype=float)


def occlusion_words(predictor, text: str, target: int) -> tuple[list[str], np.ndarray, float]:
    """Leave-one-word-out: delta_i = P(target | x) - P(target | x without word i)."""
    words = text.split()
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = predictor.predict_proba([text] + variants, clean=False)[:, target]
    return words, probs[0] - probs[1:], float(probs[0])


def deletion_test(predictor, text: str, words: list[str], scores: np.ndarray, target: int,
                  ks=(1, 2, 3), n_random: int = 10, seed: int = 0) -> dict:
    """P(target) drop after deleting top-k attributed words vs. random words."""
    rng = np.random.default_rng(seed)
    toks = text.split()
    if len(toks) != len(words):   # fall back to word list from attribution
        toks = list(words)
    base = predictor.predict_proba([text], clean=False)[0, target]
    order = np.argsort(-scores)
    res = {"base_prob": float(base)}
    for k in ks:
        if k >= len(toks):
            continue
        top = set(order[:k].tolist())
        x_top = " ".join(w for i, w in enumerate(toks) if i not in top)
        rand_sets = [set(rng.choice(len(toks), k, replace=False).tolist()) for _ in range(n_random)]
        x_rand = [" ".join(w for i, w in enumerate(toks) if i not in s) for s in rand_sets]
        p = predictor.predict_proba([x_top] + x_rand, clean=False)[:, target]
        res[f"drop_top{k}"] = float(base - p[0])
        res[f"drop_rand{k}"] = float(base - p[1:].mean())
    return res


def rank_agreement(a: np.ndarray, b: np.ndarray, k: int = 3) -> dict:
    if len(a) < 2:
        return {"spearman": None, f"top{k}_overlap": None}
    rho = spearmanr(a, b).correlation
    kk = min(k, len(a))
    ov = len(set(np.argsort(-a)[:kk]) & set(np.argsort(-b)[:kk])) / kk
    return {"spearman": None if np.isnan(rho) else float(rho), f"top{k}_overlap": float(ov)}
