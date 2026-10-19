import numpy as np

from src.evaluation.early import Conversation, aggregate, alarm_index, evaluate_alarms, prefix_curve
from src.evaluation.metrics import (bootstrap_ci, brier_score, classification_metrics,
                                    expected_calibration_error, fit_temperature, mcnemar_exact,
                                    paired_bootstrap, softmax)


def test_classification_metrics_match_hand_computation():
    y = np.array([0, 0, 1, 1, 2, 2])
    P = np.eye(3)[[0, 1, 1, 1, 2, 0]] * 0.9 + 0.1 / 3
    m = classification_metrics(y, P, ["a", "b", "c"])
    assert abs(m["accuracy"] - 4 / 6) < 1e-9
    # per-class F1: a: p=1/2,r=1/2 -> .5 ; b: p=2/3,r=1 -> .8 ; c: p=1,r=.5 -> 2/3
    assert abs(m["macro_f1"] - (0.5 + 0.8 + 2 / 3) / 3) < 1e-9
    assert m["confusion_matrix"][0] == [1, 1, 0]
    assert "roc_auc_macro_ovr" in m and "ece" in m


def test_binary_auc_and_brier():
    y = np.array([0, 1, 0, 1])
    P = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
    m = classification_metrics(y, P, ["n", "p"])
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert abs(brier_score(y, P) - np.mean([2 * 0.1**2, 2 * 0.2**2, 2 * 0.4**2, 2 * 0.3**2])) < 1e-9


def test_ece_perfectly_calibrated_is_zero():
    y = np.array([0] * 10)
    P = np.tile([1.0, 0.0], (10, 1))
    assert expected_calibration_error(y, P)["ece"] == 0.0


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.where(rng.random(300) < 0.8, y, 1 - y)
    ci = bootstrap_ci(y, p, n_boot=300)
    assert ci["lo"] <= ci["point"] <= ci["hi"]


def test_paired_tests_identical_models():
    y = np.array([0, 1] * 50)
    r = paired_bootstrap(y, y, y, n_boot=200)
    assert r["delta"] == 0.0
    assert mcnemar_exact(y, y, y)["p_value"] == 1.0


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    logits = np.stack([np.zeros(2000), np.where(y == 1, 1.0, -1.0) + rng.normal(0, 2, 2000)], 1) * 5
    T = fit_temperature(logits, y)
    assert T > 1.5


def test_softmax_rows_sum_to_one():
    assert np.allclose(softmax(np.array([[1.0, 2.0, 3.0]])).sum(), 1)


def conv(labels, scores):
    return Conversation("c", [str(i) for i in range(len(labels))], np.array(labels), np.array(scores, float))


def test_alarm_is_causal_and_delay_correct():
    c = conv([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.95])
    assert alarm_index(c, 0.5) == 2
    r = evaluate_alarms([c], 0.5)
    assert r["tp"] == 1 and r["delay_mean"] == 0
    # A future high score must not affect earlier prefix decisions
    pc = prefix_curve([c], 0.5, max_k=2)
    assert pc[0]["eventual_recall"] == 0.0 and pc[1]["eventual_recall"] == 0.0


def test_false_alarm_and_miss():
    neg = conv([0, 0], [0.7, 0.1])
    pos = conv([0, 1], [0.1, 0.2])
    r = evaluate_alarms([neg, pos], 0.5)
    assert r["fp"] == 1 and r["fn"] == 1 and r["false_alarm_rate"] == 1.0
    assert r["erde_5"] == (0.5 + 1.0) / 2      # FP cost = positive rate (0.5), FN cost = 1


def test_aggregate_policies():
    s = np.array([0.2, 0.8, 0.1])
    assert list(aggregate(s, "max")) == [0.2, 0.8, 0.8]
    assert np.allclose(aggregate(s, "ema", 0.5), [0.2, 0.5, 0.3])
