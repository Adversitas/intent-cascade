"""Production monitoring: Prometheus metrics and confidence drift.

Labels arrive late (or never) in production, so accuracy can't be watched
directly. The model's confidence distribution is available immediately and is a
useful proxy: when customers start asking about something new, or phrase old
intents differently, confidence drops and the deferral rate climbs. Drift is
measured with the Population Stability Index between the confidence histogram
recorded on the test set at training time and the recent production histogram.

PSI < 0.1: stable.  0.1-0.25: moderate shift, investigate.  > 0.25: significant shift.
"""

from __future__ import annotations

import numpy as np

CONFIDENCE_BINS = np.linspace(0.0, 1.0, 11)


def histogram(confidences: np.ndarray | list[float]) -> list[float]:
    counts, _ = np.histogram(np.asarray(confidences, dtype=float), bins=CONFIDENCE_BINS)
    total = counts.sum()
    return (counts / total).tolist() if total else [0.0] * (len(CONFIDENCE_BINS) - 1)


def psi(reference: list[float], current: list[float], eps: float = 1e-4) -> float:
    ref = np.clip(np.asarray(reference, dtype=float), eps, None)
    cur = np.clip(np.asarray(current, dtype=float), eps, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_status(value: float) -> str:
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "significant"


def drift_report(reference_hist: list[float], confidences: list[float], threshold: float) -> dict:
    conf = np.asarray(confidences, dtype=float)
    value = psi(reference_hist, histogram(conf))
    return {
        "n": int(len(conf)),
        "psi": round(value, 4),
        "status": drift_status(value),
        "mean_confidence": round(float(conf.mean()), 4) if len(conf) else None,
        "deferral_rate": round(float((conf < threshold).mean()), 4) if len(conf) else None,
        "reference_histogram": [round(x, 4) for x in reference_hist],
        "current_histogram": [round(x, 4) for x in histogram(conf)],
    }


class Metrics:
    """Prometheus collectors, on a private registry so tests can create several apps."""

    def __init__(self) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = CollectorRegistry()
        self.requests = Counter(
            "intent_requests_total",
            "Classified messages by route",
            ["route"],
            registry=self.registry,
        )
        self.confidence = Histogram(
            "intent_model_confidence",
            "Calibrated confidence of the trained model",
            buckets=[float(b) for b in CONFIDENCE_BINS[1:]],
            registry=self.registry,
        )
        self.latency = Histogram(
            "intent_latency_seconds",
            "End-to-end latency per message by route",
            ["route"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            registry=self.registry,
        )
        self.llm_tokens = Counter(
            "intent_llm_tokens_total",
            "LLM tokens used by the fallback",
            ["direction"],
            registry=self.registry,
        )
        self.psi = Gauge(
            "intent_confidence_psi",
            "PSI of recent confidence vs training reference",
            registry=self.registry,
        )

    def observe(self, decisions) -> None:
        for d in decisions:
            self.requests.labels(d.route).inc()
            self.confidence.observe(d.confidence)
            self.latency.labels(d.route).observe(d.latency_ms / 1000)
            if d.llm_input_tokens:
                self.llm_tokens.labels("input").inc(d.llm_input_tokens)
                self.llm_tokens.labels("output").inc(d.llm_output_tokens)

    def render(self) -> tuple[bytes, str]:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST
