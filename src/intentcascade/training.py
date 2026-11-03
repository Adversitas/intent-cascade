"""Train → calibrate → evaluate → register."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .calibration import TemperatureScaler, softmax
from .data import DATA_DIR, Example, LabelSet, load_csv, stratified_split
from .metrics import (
    accuracy,
    aurc,
    coverage_at_accuracy,
    expected_calibration_error,
    macro_f1,
    nll,
    threshold_table,
    top_k_accuracy,
)
from .models import TrainConfig, build_model, train_config_dict
from .monitoring import histogram
from .registry import new_version_name, save_bundle

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def evaluate_probs(probs: np.ndarray, y: np.ndarray) -> dict:
    return {
        "n": int(len(y)),
        "accuracy": round(accuracy(y, probs.argmax(1)), 4),
        "macro_f1": round(macro_f1(y, probs.argmax(1)), 4),
        "top5_accuracy": round(top_k_accuracy(probs, y, 5), 4),
        "top10_accuracy": round(top_k_accuracy(probs, y, 10), 4),
        "nll": round(nll(probs, y), 4),
        "ece": round(expected_calibration_error(probs, y), 4),
        "aurc": round(aurc(probs, y), 4),
        "coverage_at_95_accuracy": round(coverage_at_accuracy(probs, y, 0.95), 4),
        "coverage_at_98_accuracy": round(coverage_at_accuracy(probs, y, 0.98), 4),
        "thresholds": [
            {k: round(v, 4) for k, v in row.items()}
            for row in threshold_table(probs, y, THRESHOLDS)
        ],
        "confidence_histogram": [round(x, 4) for x in histogram(probs.max(1))],
    }


def train_and_register(
    kind: str = "fasttext",
    train_path: Path = DATA_DIR / "train.csv",
    test_path: Path = DATA_DIR / "test.csv",
    registry: Path = Path("models"),
    config: TrainConfig | None = None,
    model_kwargs: dict | None = None,
    log=print,
) -> tuple[Path, dict]:
    cfg = config or TrainConfig()
    all_train = load_csv(train_path)
    test = load_csv(test_path)
    train, val = stratified_split(all_train, val_fraction=0.1, seed=cfg.seed)
    labels = LabelSet([e.label for e in all_train])
    log(f"train {len(train)}  val {len(val)}  test {len(test)}  intents {len(labels)}")

    model = build_model(kind, labels, **(model_kwargs or {}))
    t0 = time.perf_counter()
    history = model.fit(train, val, cfg, log=log)
    train_seconds = time.perf_counter() - t0

    y_val = np.array(labels.encode([e.label for e in val]))
    val_logits = model.logits([e.text for e in val])
    scaler = TemperatureScaler().fit(val_logits, y_val)
    log(f"temperature {scaler.temperature:.3f}")

    metrics = {
        "validation": evaluate_probs(scaler(val_logits), y_val),
        "validation_uncalibrated": evaluate_probs(softmax(val_logits), y_val),
    }
    metrics["test"], metrics["test_uncalibrated"], latency = evaluate_split(model, scaler, test)
    metrics["test"]["latency_ms_per_message"] = latency

    name = new_version_name(kind)
    card = {
        "name": name,
        "kind": kind,
        "model_config": model._config(),
        "train_config": train_config_dict(cfg),
        "data": {
            "train": str(train_path),
            "test": str(test_path),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_labels": len(labels),
        },
        "temperature": scaler.temperature,
        "train_seconds": round(train_seconds, 1),
        "history": history.epochs,
        "best_epoch": history.best_epoch,
    }
    path = save_bundle(registry, name, model, scaler, metrics, card)
    return path, metrics


def evaluate_split(model, scaler: TemperatureScaler, examples: list[Example]):
    y = np.array(model.labels.encode([e.label for e in examples]))
    texts = [e.text for e in examples]
    t0 = time.perf_counter()
    logits = model.logits(texts)
    latency = (time.perf_counter() - t0) * 1000 / max(len(texts), 1)
    return evaluate_probs(scaler(logits), y), evaluate_probs(softmax(logits), y), round(latency, 3)


def tfidf_baseline(train: list[Example], test: list[Example], labels: LabelSet) -> dict:
    """TF-IDF + logistic regression: the classical baseline any neural model has to beat."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, make_pipeline

    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ]
    )
    clf = make_pipeline(features, LogisticRegression(C=20, max_iter=2000))
    tr, val = stratified_split(train, 0.1, seed=0)
    clf.fit([e.text for e in tr], labels.encode([e.label for e in tr]))
    val_logits = clf.decision_function([e.text for e in val])
    scaler = TemperatureScaler().fit(val_logits, np.array(labels.encode([e.label for e in val])))
    y = np.array(labels.encode([e.label for e in test]))
    t0 = time.perf_counter()
    logits = clf.decision_function([e.text for e in test])
    latency = (time.perf_counter() - t0) * 1000 / len(test)
    out = evaluate_probs(scaler(logits), y)
    out["latency_ms_per_message"] = round(latency, 3)
    return out
