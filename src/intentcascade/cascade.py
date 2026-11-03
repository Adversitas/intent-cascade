"""Confidence-based routing between the trained model, the LLM and a human.

    message ──► trained model ──► confidence ≥ threshold? ──yes──► answer (route="model")
                                        │ no
                                        ▼
                     top-k candidates + nearest labelled examples
                                        │
                                        ▼
                                LLM picks a candidate ──► answer (route="llm")
                                        │ "none_of_these", or LLM unavailable
                                        ▼
                                  route="human"

The threshold is the single knob that trades accuracy against LLM cost and
latency; ``metrics.threshold_table`` and the ``evaluate`` command show how much
traffic each threshold sends to the LLM and how accurate the model is on what
it keeps.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .calibration import TemperatureScaler
from .llm import NONE_LABEL, ClaudeIntentClassifier
from .models import _TorchClassifier
from .vectorstore import NumpyVectorStore, VectorStore

log = logging.getLogger(__name__)


@dataclass
class Decision:
    text: str
    label: str | None
    route: str  # "model" | "llm" | "human"
    confidence: float
    model_label: str
    candidates: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    reason: str = ""


class Cascade:
    def __init__(
        self,
        model: _TorchClassifier,
        scaler: TemperatureScaler,
        store: VectorStore | None = None,
        llm: ClaudeIntentClassifier | None = None,
        threshold: float = 0.6,
        n_candidates: int = 10,
        n_examples: int = 12,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.store = store if store is not None else NumpyVectorStore()
        self.llm = llm
        self.threshold = threshold
        self.n_candidates = n_candidates
        self.n_examples = n_examples

    def index_examples(self, texts: list[str], labels: list[str]) -> None:
        """Load labelled examples into the vector store for few-shot retrieval."""
        self.store.add(texts, labels, self.model.embed(texts))

    def predict(self, texts: list[str]) -> list[Decision]:
        start = time.perf_counter()
        probs = self.scaler(self.model.logits(texts))
        model_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
        names = self.model.labels.names
        decisions = []
        for text, p in zip(texts, probs):
            t0 = time.perf_counter()
            top = np.argsort(-p)[: self.n_candidates]
            d = Decision(
                text=text,
                label=names[top[0]],
                route="model",
                confidence=float(p[top[0]]),
                model_label=names[top[0]],
                candidates=[names[i] for i in top],
            )
            if d.confidence < self.threshold:
                self._defer(d)
            d.latency_ms = model_ms + (time.perf_counter() - t0) * 1000
            decisions.append(d)
        return decisions

    def _defer(self, d: Decision) -> None:
        if self.llm is None:
            d.label, d.route, d.reason = None, "human", "low confidence; no LLM configured"
            return
        examples = self.store.search(self.model.embed([d.text])[0], k=self.n_examples)
        # Retrieved neighbours can surface the right intent even when the model ranked
        # it outside its top-k, so their labels are added to the candidate list.
        candidates = list(dict.fromkeys(d.candidates + [ex.label for ex in examples]))
        d.candidates = candidates
        try:
            result = self.llm.classify(d.text, candidates, examples)
        except Exception as exc:  # any API or parsing failure degrades to a human hand-off
            log.warning("LLM fallback failed: %s", exc)
            d.label, d.route, d.reason = None, "human", f"LLM error: {exc}"
            return
        d.llm_input_tokens, d.llm_output_tokens = result.input_tokens, result.output_tokens
        if result.label == NONE_LABEL:
            d.label, d.route, d.reason = None, "human", "LLM found no matching intent"
        else:
            d.label, d.route = result.label, "llm"
