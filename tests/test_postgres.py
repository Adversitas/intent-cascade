"""Integration tests against a real PostgreSQL + pgvector (run in CI's `postgres` job).

Skipped unless TEST_POSTGRES_URL is set, e.g.
    docker compose up -d db
    TEST_POSTGRES_URL=postgresql://intent:intent@localhost:5432/intent pytest tests/test_postgres.py
"""

import os

import numpy as np
import pytest

from intentcascade.cascade import Decision
from intentcascade.db import PredictionLog

URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_POSTGRES_URL not set")


def test_pgvector_store_matches_numpy_store(trained, small_data):
    from intentcascade.vectorstore import NumpyVectorStore, PgVectorStore

    model, _ = trained
    train, test, _ = small_data
    texts, labels = [e.text for e in train], [e.label for e in train]
    vectors = model.embed(texts)

    pg = PgVectorStore(URL, dim=vectors.shape[1], table="test_examples")
    pg.clear()
    pg.add(texts, labels, vectors)
    mem = NumpyVectorStore()
    mem.add(texts, labels, vectors)
    assert len(pg) == len(mem) == len(texts)

    # HNSW is approximate, so compare label overlap rather than exact order.
    for q in model.embed([e.text for e in test[:10]]):
        pg_labels = [n.label for n in pg.search(q, k=5)]
        mem_labels = [n.label for n in mem.search(q, k=5)]
        assert pg_labels[0] == mem_labels[0]
        assert np.isclose(pg.search(q, k=1)[0].score, mem.search(q, k=1)[0].score, atol=1e-4)


def test_prediction_log_on_postgres():
    log = PredictionLog(URL.replace("postgresql://", "postgresql+psycopg://"))
    d = Decision(
        text="hi", label="card_arrival", route="model", confidence=0.9, model_label="card_arrival"
    )
    [pid] = log.record([d], "pg-test")
    assert log.add_feedback(pid, "card_arrival")
    assert log.stats()["routes"]["model"]["count"] >= 1
    assert 0.9 in log.confidences("pg-test")
