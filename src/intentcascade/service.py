"""Assemble a ready-to-serve cascade from a registry bundle and environment settings."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from .cascade import Cascade
from .data import DATA_DIR, load_csv
from .registry import Bundle, load_bundle

log = logging.getLogger(__name__)


@dataclass
class Service:
    bundle: Bundle
    cascade: Cascade

    @property
    def version(self) -> str:
        return self.bundle.name

    @property
    def reference_histogram(self) -> list[float]:
        return self.bundle.metrics["test"]["confidence_histogram"]


def build_service(
    registry: str | Path | None = None,
    model_name: str | None = None,
    threshold: float | None = None,
    examples_path: str | Path | None = None,
    vector_db_url: str | None = None,
    use_llm: bool | None = None,
) -> Service:
    registry = registry or os.environ.get("INTENT_CASCADE_MODELS", "models")
    model_name = model_name or os.environ.get("INTENT_CASCADE_MODEL") or None
    if threshold is None:
        threshold = float(os.environ.get("INTENT_CASCADE_THRESHOLD", "0.6"))
    examples_path = examples_path or os.environ.get(
        "INTENT_CASCADE_EXAMPLES", str(DATA_DIR / "train.csv")
    )
    vector_db_url = vector_db_url or os.environ.get("VECTOR_DB_URL") or None
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))

    bundle = load_bundle(registry, model_name)
    store = None
    if vector_db_url:
        from .vectorstore import PgVectorStore

        dim = bundle.model.embed(["probe"]).shape[1]
        # One table per model version: embeddings from different models aren't comparable.
        table = "examples_" + "".join(c if c.isalnum() else "_" for c in bundle.name).lower()
        store = PgVectorStore(vector_db_url, dim=dim, table=table)

    llm = None
    if use_llm:
        from .llm import ClaudeIntentClassifier

        llm = ClaudeIntentClassifier()
    cascade = Cascade(bundle.model, bundle.scaler, store=store, llm=llm, threshold=threshold)

    if len(cascade.store) == 0:
        examples = load_csv(examples_path)
        cascade.index_examples([e.text for e in examples], [e.label for e in examples])
        log.info("indexed %d labelled examples", len(examples))
    log.info(
        "serving %s threshold=%.2f llm=%s vector_store=%s",
        bundle.name,
        threshold,
        bool(llm),
        type(cascade.store).__name__,
    )
    return Service(bundle=bundle, cascade=cascade)
