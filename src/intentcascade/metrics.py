"""Classification, calibration and selective-prediction metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    p = probs[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Gap between stated confidence and observed accuracy, averaged over confidence bins."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def top_k_accuracy(probs: np.ndarray, y_true: np.ndarray, k: int) -> float:
    top = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, top)]))


def risk_coverage(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy on the most-confident fraction of examples, for every coverage level.

    Sorting by confidence and keeping the top c% simulates a model that answers
    only when confident and defers the rest. A well-ranked confidence score
    gives accuracy that rises steeply as coverage falls.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    order = np.argsort(-conf, kind="stable")
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    coverage = np.arange(1, len(order) + 1) / len(order)
    return coverage, cum_acc


def aurc(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    _, acc = risk_coverage(probs, y_true)
    return float(np.mean(1.0 - acc))


def threshold_table(
    probs: np.ndarray, y_true: np.ndarray, thresholds: list[float]
) -> list[dict[str, float]]:
    """For each confidence threshold: how much the model keeps, and how accurate that slice is."""
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    rows = []
    for t in thresholds:
        keep = conf >= t
        rows.append(
            {
                "threshold": t,
                "coverage": float(keep.mean()),
                "deferred": float(1 - keep.mean()),
                "accuracy_kept": float(correct[keep].mean()) if keep.any() else float("nan"),
                "accuracy_deferred_if_model_kept_them": (
                    float(correct[~keep].mean()) if (~keep).any() else float("nan")
                ),
            }
        )
    return rows


def coverage_at_accuracy(probs: np.ndarray, y_true: np.ndarray, target: float) -> float:
    """Largest fraction of traffic the model can handle alone at >= `target` accuracy."""
    coverage, acc = risk_coverage(probs, y_true)
    ok = np.where(acc >= target)[0]
    return float(coverage[ok.max()]) if len(ok) else 0.0
