"""Inference, explanation, export and app-startup tests (tiny random model; no real data needed)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

TINY = "hf-internal-testing/tiny-random-BertModel"
TINY_REV = "fc08ad9cc33be9aef4f55cc80e16ef5ae3d5981c"


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    """Random-initialised tiny BERT saved locally, wrapped as a SACENet checkpoint."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from src.models.transformer import SACENetClassifier

    root = tmp_path_factory.mktemp("tiny")
    try:
        cfg = AutoConfig.from_pretrained(TINY, revision=TINY_REV)
        tok = AutoTokenizer.from_pretrained(TINY, revision=TINY_REV)
    except Exception as e:  # offline machine without the cached tiny model
        pytest.skip(f"tiny test model unavailable: {e}")
    enc_dir = root / "encoder"
    torch.manual_seed(0)
    AutoModel.from_config(cfg).save_pretrained(enc_dir)
    tok.save_pretrained(enc_dir)
    labels = ["not_cyberbullying", "age", "gender", "religion", "mockery", "abusive", "offensive"]
    m = SACENetClassifier(str(enc_dir), {"main": 7}, freeze_word_embeddings=True)
    assert not m.encoder.get_input_embeddings().weight.requires_grad
    out = root / "model"
    out.mkdir()
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in m.state_dict().items()}, out / "weights.pt")
    tok.save_pretrained(out)
    (out / "sacenet_config.json").write_text(json.dumps({
        "model_name": str(enc_dir), "revision": None, "heads": {"main": 7}, "labels": labels, "fusion_dim": 0,
        "dropout": 0.1, "max_len": 32, "temperature": 1.5}))
    return out


def test_predictor_outputs_valid_probabilities(tiny_ckpt):
    from src.inference.predictor import Predictor
    p = Predictor(tiny_ckpt, device="cpu")
    P = p.predict_proba(["kya bakwas hai yeh", "nice match bhai 😂", ""])
    assert P.shape == (3, 7)
    assert np.allclose(P.sum(1), 1, atol=1e-5)
    # temperature is applied: calibrated probs are flatter than raw ones
    raw = Predictor(tiny_ckpt, device="cpu", calibrated=False).predict_proba(["kya bakwas hai yeh"])
    assert P[0].max() <= raw[0].max() + 1e-6


def test_fusion_model_forward_shapes():
    from transformers import AutoConfig, AutoModel
    import tempfile
    from src.models.transformer import SACENetClassifier
    try:
        cfg = AutoConfig.from_pretrained(TINY, revision=TINY_REV)
    except Exception as e:
        pytest.skip(str(e))
    with tempfile.TemporaryDirectory() as d:
        AutoModel.from_config(cfg).save_pretrained(d)
        m = SACENetClassifier(d, {"main": 7, "aux": 2}, fusion_dim=5)
        ids = torch.randint(5, 50, (2, 6))
        am = torch.ones_like(ids)
        assert m(ids, am, "main", fusion_feats=torch.randn(2, 5)).shape == (2, 7)
        assert m(ids, am, "aux").shape == (2, 2)
        with pytest.raises(ValueError):
            m(ids, am, "main")


def test_shap_explanation_and_checks(tiny_ckpt):
    from src.explainability.explain import deletion_test, occlusion_words, rank_agreement, shap_explain, to_words
    from src.inference.predictor import Predictor
    p = Predictor(tiny_ckpt, device="cpu")
    e = shap_explain(p, ["yeh banda bilkul pagal hai"], max_evals=40)[0]
    assert len(e["tokens"]) == len(e["values"]) and e["seconds"] > 0
    # SHAP efficiency: base + sum(values) ~= f(x) for the explained class
    assert abs(e["base_value"] + sum(e["values"]) - e["prob"][e["target"]]) < 0.05
    words, wv = to_words(e["tokens"], e["values"])
    assert words == ["yeh", "banda", "bilkul", "pagal", "hai"]
    ow, od, base = occlusion_words(p, e["text"], e["target"])
    assert len(ow) == 5 and len(od) == 5
    d = deletion_test(p, e["text"], words, wv, e["target"], ks=(1,), n_random=3)
    assert "drop_top1" in d and "drop_rand1" in d
    ra = rank_agreement(np.array([3.0, 2, 1]), np.array([3.0, 2, 1]))
    assert ra["spearman"] == pytest.approx(1.0) and ra["top3_overlap"] == 1.0


def test_figure_and_table_export(tmp_path):
    import matplotlib.pyplot as plt
    from src.reporting.style import save
    from src.reporting.tables import write
    reg = []
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    save(fig, tmp_path, "figx", pd.DataFrame({"a": [1]}), reg, "cap")
    assert all((tmp_path / f"figx.{s}").exists() for s in ("png", "pdf", "csv"))
    write(pd.DataFrame({"Model": ["m_1"], "F1": [0.12345]}), tmp_path, "tabx", "Caption & 50%", reg)
    tex = (tmp_path / "tabx.tex").read_text()
    assert r"\toprule" in tex and r"m\_1" in tex and "0.123" in tex and r"\&" in tex
    assert "| m_1 | 0.123 |" in (tmp_path / "tabx.md").read_text()
    assert len(reg) == 2


def test_jsonl_roundtrip_with_numpy(tmp_path):
    from src.utils.io import load_jsonl, save_jsonl
    save_jsonl([{"id": "a", "p": np.array([0.1, 0.9]), "y": np.int64(1)}], tmp_path / "x.jsonl")
    assert load_jsonl(tmp_path / "x.jsonl") == [{"id": "a", "p": [0.1, 0.9], "y": 1}]


def test_app_starts_and_reports_missing_model(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SACENET_CB_MODEL", "/nonexistent/model")
    monkeypatch.setenv("SACENET_SARC_MODEL", "/nonexistent/model")
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.text_area[0].input("kya bakwas hai").run()
    at.button[0].click().run()
    assert not at.exception
    assert any("not found" in e.value for e in at.error)
