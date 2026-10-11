import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from landslide.metrics import BinaryMeter  # noqa: E402


def _meter(prob, target):
    m = BinaryMeter()
    m.update(np.asarray(prob, dtype=np.float64), np.asarray(target))
    return m.compute()


def test_perfect_prediction():
    t = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    r = _meter(np.where(t == 1, 0.999, 0.001), t)
    assert r["iou_landslide"] > 0.999 and r["f1"] > 0.999 and r["pr_auc"] > 0.99 and r["ece"] < 0.01


def test_iou_f1_values_by_hand():
    # tp=2 fp=1 fn=1 tn=4 -> IoU 2/4, precision 2/3, recall 2/3, F1 2/3
    prob = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
    tgt = [1, 1, 0, 1, 0, 0, 0, 0]
    r = _meter(prob, tgt)
    assert abs(r["iou_landslide"] - 0.5) < 1e-6 and abs(r["f1"] - 2 / 3) < 1e-6
    assert abs(r["miou"] - (0.5 + 4 / 6) / 2) < 1e-6


def test_overconfident_wrong_predictions_have_high_ece():
    t = np.zeros(1000, dtype=int)
    r = _meter(np.full(1000, 0.99), t)            # always confident landslide, always wrong
    assert r["ece"] > 0.9


def test_reliability_bins_reproduce_ece():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 50000)
    t = (rng.uniform(0, 1, p.size) < np.clip(p * 0.8 + 0.1, 0, 1)).astype(int)     # deliberately miscalibrated
    m = BinaryMeter(); m.update(p, t)
    rel = m.reliability()
    tot = sum(rel["count"])
    ece = sum(c / tot * abs(a - f) for c, a, f in zip(rel["count"], rel["accuracy"], rel["confidence"]) if c)
    assert abs(ece - m.compute()["ece"]) < 1e-9 and tot == p.size


def test_calibrated_predictions_have_low_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, 200000)
    t = (rng.uniform(0, 1, p.size) < p).astype(int)    # outcome drawn with probability p -> perfectly calibrated
    assert _meter(p, t)["ece"] < 0.01
