"""Nearest-neighbour search over labelled examples.

When the cascade defers a message to the LLM, it retrieves the most similar
labelled messages and shows them as few-shot examples. This is retrieval-
augmented classification: the LLM sees how *this* bank's customers phrase each
intent instead of guessing from the label name alone.

Two backends share one interface:

- ``NumpyVectorStore``: exact cosine search in memory. Default; no services needed.
- ``PgVectorStore``: PostgreSQL with the pgvector extension and an HNSW index,
  for when the example bank is shared between replicas or grows past memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Neighbour:
    text: str
    label: str
    score: float


class VectorStore(Protocol):
    def add(self, texts: list[str], labels: list[str], vectors: np.ndarray) -> None: ...

    def search(self, vector: np.ndarray, k: int = 8) -> list[Neighbour]: ...

    def __len__(self) -> int: ...


class NumpyVectorStore:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.labels: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def add(self, texts: list[str], labels: list[str], vectors: np.ndarray) -> None:
        if not (len(texts) == len(labels) == len(vectors)):
            raise ValueError("texts, labels and vectors must have the same length")
        vectors = np.asarray(vectors, dtype=np.float32)
        self.vectors = vectors if not len(self.texts) else np.vstack([self.vectors, vectors])
        self.texts += list(texts)
        self.labels += list(labels)

    def search(self, vector: np.ndarray, k: int = 8) -> list[Neighbour]:
        if not self.texts:
            return []
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Neighbour(self.texts[i], self.labels[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self.texts)


class PgVectorStore:
    """pgvector-backed store. Vectors are L2-normalised, so cosine order = dot-product order."""

    def __init__(self, dsn: str, dim: int, table: str = "examples") -> None:
        import psycopg
        from pgvector.psycopg import register_vector

        if not table.isidentifier():
            raise ValueError(f"invalid table name {table!r}")
        self.table, self.dim = table, dim
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            f" id bigserial PRIMARY KEY, text text NOT NULL, label text NOT NULL,"
            f" embedding vector({dim}) NOT NULL)"
        )
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw ON {table}"
            f" USING hnsw (embedding vector_cosine_ops)"
        )

    def add(self, texts: list[str], labels: list[str], vectors: np.ndarray) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (text, label, embedding) VALUES (%s, %s, %s)",
                [
                    (t, lab, np.asarray(v, dtype=np.float32))
                    for t, lab, v in zip(texts, labels, vectors)
                ],
            )

    def search(self, vector: np.ndarray, k: int = 8) -> list[Neighbour]:
        rows = self.conn.execute(
            f"SELECT text, label, 1 - (embedding <=> %s) AS score FROM {self.table}"
            f" ORDER BY embedding <=> %s LIMIT %s",
            (np.asarray(vector, dtype=np.float32),) * 2 + (k,),
        ).fetchall()
        return [Neighbour(t, lab, float(s)) for t, lab, s in rows]

    def clear(self) -> None:
        self.conn.execute(f"TRUNCATE {self.table}")

    def __len__(self) -> int:
        return self.conn.execute(f"SELECT count(*) FROM {self.table}").fetchone()[0]
