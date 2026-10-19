"""Data pipeline tests. All texts here are SYNTHETIC fixtures for software testing only."""
import numpy as np
import pytest

from src.data import loaders
from src.data.quality import (connected_groups, exact_duplicate_groups, label_conflicts, leakage_report,
                              near_duplicate_pairs)
from src.data.schema import MC_LABELS, MC_TO_BINARY, Record
from src.data.splits import group_split_by_key, stratified_group_split
from src.data.text import clean_text, normalised_key, script_of
from src.training.data import class_weights, encode_label


def rec(i, text, lab="age"):
    return Record(id=f"t-{i}", text=clean_text(text), text_raw=text, source="test", language="hi-en",
                  script=script_of(text), label_orig=lab, cb_type=lab)


def test_clean_preserves_code_mixing_case_and_punctuation():
    t = clean_text("Wah BHAI!!! kya baat hai 😂 @rahul https://x.co/a")
    assert "BHAI!!!" in t and "kya baat hai" in t
    assert ":face_with_tears_of_joy:" in t
    assert "@user" in t and "URL" in t and "rahul" not in t


def test_clean_caps_elongation_but_keeps_signal():
    assert clean_text("yaaaaaaar") == "yaaar"


def test_script_detection():
    assert script_of("kya hua") == "roman"
    assert script_of("क्या हुआ") == "devanagari"
    assert script_of("क्या hua") == "mixed"
    assert script_of("123 !!") == "other"


def test_normalised_key_collapses_variants():
    assert normalised_key("Bhaaai!!  kya") == normalised_key("bhai kya")


def test_binary_mapping_is_total_and_correct():
    assert set(MC_TO_BINARY) == set(MC_LABELS)
    assert MC_TO_BINARY["not_cyberbullying"] == 0
    assert all(v == 1 for k, v in MC_TO_BINARY.items() if k != "not_cyberbullying")


def test_encode_label_mc7_and_conda():
    assert encode_label("mc7", "not_cyberbullying") == 0
    assert encode_label("mc7", "offensive") == MC_LABELS.index("offensive")
    assert encode_label("conda", "E") == 1 and encode_label("conda", "I") == 1
    assert encode_label("conda", "O") == 0 and encode_label("conda", "A") == 0


def test_unknown_label_raises(tmp_path):
    d = tmp_path / "mc"
    d.mkdir()
    (d / "train.csv").write_text("tweets,cyberbullying_type\nhello,not_a_label\n")
    (d / "test.csv").write_text("tweets,cyberbullying_type\nhi,age\n")
    with pytest.raises(loaders.DataValidationError):
        loaders.load_mc_hinglish(d)


def test_swami_block_parser(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    a, b, c, z = "866871160725794816", "880356789358743553", "877751493889105920", "999999999999999999"
    # tweets: blank-line separated, one tweet spans two lines; truth: alternating lines (real layout)
    (d / "Sarcasm_tweets.txt").write_text(f"{a}\nwah kya baat\n\n{b}\nnormal\ntweet 2019\n\n{c}\norphan\n")
    (d / "Sarcasm_tweet_truth.txt").write_text(f"{a}\nYES\n{b}\nNO\n{z}\nNO\n")
    recs = {r.meta["tweet_id"]: r for r in loaders.load_swami_sarcasm(d)}
    assert {k: r.sarcasm for k, r in recs.items()} == {a: 1, b: 0}
    assert recs[b].text_raw == "normal tweet 2019"


def test_duplicates_and_conflicts():
    rs = [rec(0, "same text", "age"), rec(1, "same text", "gender"), rec(2, "SAME   text!!", "age"),
          rec(3, "different")]
    assert len(exact_duplicate_groups(rs, "raw")) == 1
    norm = exact_duplicate_groups(rs, "norm")
    assert len(norm) == 1 and len(norm[0]) == 3
    assert len(label_conflicts(rs, norm, "cb_type")) == 1


def test_near_duplicates_detected():
    a = ["yeh banda bilkul pagal hai yaar", "kal match dekhne chalein"]
    b = ["yeh banda bilkul pagal hai yaaar", "aaj mausam accha hai"]
    pairs = near_duplicate_pairs(a, b, threshold=0.8)
    assert (0, 0) in {(i, j) for i, j, _ in pairs}
    assert all(j != 1 for _, j, _ in pairs)


def test_connected_groups():
    g = connected_groups(5, [(0, 1), (1, 2)])
    assert g[0] == g[1] == g[2] and len(set(g)) == 3


def test_group_split_keeps_groups_together_and_stratifies():
    rng = np.random.default_rng(0)
    labels = list(rng.integers(0, 3, 600))
    groups = np.repeat(np.arange(300), 2)       # pairs that must stay together
    mask = stratified_group_split(labels, groups, 0.1, seed=1)
    for g in range(300):
        assert mask[2 * g] == mask[2 * g + 1]
    assert 0.05 < mask.mean() < 0.15


def test_group_split_by_key_is_deterministic_and_disjoint():
    keys = [str(i % 50) for i in range(500)]
    a = group_split_by_key(keys, {"train": 0.8, "val": 0.2}, seed=3)
    b = group_split_by_key(keys, {"train": 0.8, "val": 0.2}, seed=3)
    assert a == b and set(a.values()) == {"train", "val"}


def test_leakage_report_detects_overlap():
    tr = [rec(0, "exactly this"), rec(1, "unique train text here")]
    te = [rec(2, "exactly this"), rec(3, "completely other content")]
    rep = leakage_report({"train": tr, "test": te})["train__test"]
    assert rep["exact"] == 1 and rep["normalised"] == 1


def test_class_weights_from_training_counts():
    w = class_weights(np.array([0, 0, 0, 1]), 2)
    assert w[1] > w[0] and abs(float(w.mean()) - 1) < 1e-6


def test_swami_collection_hashtags_removed_from_model_input(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    tid = "866871160725794816"
    (d / "Sarcasm_tweets.txt").write_text(f"{tid}\nwah kya baat hai #sarcasm #irony\n")
    (d / "Sarcasm_tweet_truth.txt").write_text(f"{tid}\nYES\n")
    r = loaders.load_swami_sarcasm(d)[0]
    assert "#" not in r.text and "sarcasm" not in r.text.lower()
    assert "#sarcasm" in r.text_raw
