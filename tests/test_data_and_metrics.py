from collections import Counter

import numpy as np
import pytest

from intentcascade.calibration import TemperatureScaler, softmax
from intentcascade.data import DATA_DIR, Example, LabelSet, load_csv, stratified_split
from intentcascade.metrics import (
    aurc,
    coverage_at_accuracy,
    expected_calibration_error,
    risk_coverage,
    threshold_table,
    top_k_accuracy,
)
from intentcascade.monitoring import drift_report, histogram, psi


def test_bundled_dataset_shape():
    train, test = load_csv(DATA_DIR / "train.csv"), load_csv(DATA_DIR / "test.csv")
    assert len(train) == 10003 and len(test) == 3080
    assert len({e.label for e in train}) == 77
    assert {e.label for e in test} <= {e.label for e in train}


def test_stratified_split_keeps_every_label_in_both_sides():
    data = [Example(f"{lab} {i}", lab) for lab in "abc" for i in range(20)]
    train, val = stratified_split(data, 0.25, seed=1)
    assert Counter(e.label for e in val) == {"a": 5, "b": 5, "c": 5}
    assert len(train) + len(val) == len(data)
    assert not {e.text for e in train} & {e.text for e in val}


def test_labelset_roundtrip():
    labels = LabelSet(["b", "a", "b"])
    assert labels.names == ["a", "b"]
    assert labels.decode(labels.encode(["b", "a"])) == ["b", "a"]


def test_load_csv_rejects_wrong_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("sentence,intent\nhi,greet\n")
    with pytest.raises(ValueError):
        load_csv(p)


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    n, k = 2000, 5
    y = rng.integers(0, k, n)
    # Logits that pick the right class 70% of the time but with huge margins.
    pred = np.where(rng.random(n) < 0.7, y, rng.integers(0, k, n))
    logits = rng.normal(0, 1, (n, k))
    logits[np.arange(n), pred] += 12
    scaler = TemperatureScaler().fit(logits, y)
    assert scaler.temperature > 2
    before = expected_calibration_error(softmax(logits), y)
    after = expected_calibration_error(scaler(logits), y)
    assert after < before / 2
    # Scaling never changes the predicted class.
    assert (scaler(logits).argmax(1) == logits.argmax(1)).all()


def test_selective_metrics_on_known_case():
    # Confident and right on the first two, unsure and wrong on the last two.
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.55, 0.45]])
    y = np.array([0, 0, 0, 1])
    coverage, acc = risk_coverage(probs, y)
    assert coverage.tolist() == [0.25, 0.5, 0.75, 1.0]
    assert acc[:2].tolist() == [1.0, 1.0] and acc[-1] == 0.5
    assert coverage_at_accuracy(probs, y, 0.99) == 0.5
    assert aurc(probs, y) == pytest.approx(np.mean(1 - acc))
    row = threshold_table(probs, y, [0.7])[0]
    assert row["coverage"] == 0.5 and row["accuracy_kept"] == 1.0
    assert row["accuracy_deferred_if_model_kept_them"] == 0.0
    assert top_k_accuracy(probs, y, 2) == 1.0


def test_psi_detects_shift():
    ref = histogram(np.linspace(0.5, 1.0, 1000))
    assert psi(ref, histogram(np.linspace(0.5, 1.0, 500))) < 0.01
    shifted = drift_report(ref, list(np.linspace(0.1, 0.6, 500)), threshold=0.6)
    assert shifted["status"] == "significant"
    assert shifted["deferral_rate"] > 0.9
