"""Trainable intent classifiers.

Every classifier exposes the same small surface:

- ``logits(texts)``   raw class scores, shape (n, n_classes)
- ``embed(texts)``    an L2-normalised representation used for nearest-neighbour
                      lookups (the vector store and the LLM's few-shot examples)
- ``save(dir)`` / ``load(dir)``

Two PyTorch models are provided:

``FastTextClassifier``
    Hashed word, bigram and character n-gram embeddings averaged into one
    vector (an ``nn.EmbeddingBag``), followed by a linear layer. Trains from
    scratch in seconds on a CPU and needs no pretrained weights, so it also runs
    in CI. Character n-grams make it robust to typos, which are common in
    support messages.

``EmbeddingMLPClassifier``
    A frozen sentence-transformer encoder with a trained MLP head. Higher
    accuracy, at the cost of a ~130 MB pretrained model and slower inference.

Probabilities are produced by ``calibration.TemperatureScaler`` rather than a
bare softmax, because the cascade routes on confidence and an overconfident
model would keep requests it gets wrong.
"""

from __future__ import annotations

import json
import re
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import Example, LabelSet

_TOKEN = re.compile(r"[a-z0-9']+|[?!£$€]")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _bucket(feature: str, n_buckets: int) -> int:
    # crc32 rather than hash(): Python's string hash is salted per process,
    # which would make a saved model unusable after a restart.
    return zlib.crc32(feature.encode("utf-8")) % n_buckets


def featurize(text: str, n_buckets: int, char_ngrams: tuple[int, int] = (3, 5)) -> list[int]:
    words = tokenize(text)
    feats = [f"w:{w}" for w in words]
    feats += [f"b:{a}_{b}" for a, b in zip(words, words[1:])]
    lo, hi = char_ngrams
    for w in words:
        padded = f"<{w}>"
        for n in range(lo, hi + 1):
            feats += [f"c:{padded[i : i + n]}" for i in range(len(padded) - n + 1)]
    return [_bucket(f, n_buckets) for f in feats] or [0]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-3
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 0
    label_smoothing: float = 0.0


@dataclass
class TrainHistory:
    epochs: list[dict] = field(default_factory=list)
    best_epoch: int = 0


def _batches(n: int, batch_size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


class _TorchClassifier:
    """Shared training loop: AdamW, early stopping on validation loss, best-weights restore."""

    kind = ""
    module: nn.Module
    labels: LabelSet

    def _optimizers(self, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
        return [
            torch.optim.AdamW(self.module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        ]

    def _forward(self, idx: np.ndarray, features) -> torch.Tensor:
        raise NotImplementedError

    def _prepare(self, texts: list[str]):
        raise NotImplementedError

    def fit(
        self,
        train: list[Example],
        val: list[Example],
        config: TrainConfig | None = None,
        log=print,
    ) -> TrainHistory:
        cfg = config or TrainConfig()
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        x_train = self._prepare([e.text for e in train])
        x_val = self._prepare([e.text for e in val])
        y_train = torch.tensor(self.labels.encode([e.label for e in train]))
        y_val = torch.tensor(self.labels.encode([e.label for e in val]))

        optimizers = self._optimizers(cfg)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        history = TrainHistory()
        best_loss, best_state, bad_epochs = float("inf"), None, 0

        for epoch in range(1, cfg.epochs + 1):
            self.module.train()
            total = 0.0
            for idx in _batches(len(train), cfg.batch_size, rng):
                for opt in optimizers:
                    opt.zero_grad()
                loss = loss_fn(self._forward(idx, x_train), y_train[idx])
                loss.backward()
                for opt in optimizers:
                    opt.step()
                total += loss.item() * len(idx)

            self.module.eval()
            with torch.no_grad():
                val_logits = self._forward(np.arange(len(val)), x_val)
                val_loss = nn.functional.cross_entropy(val_logits, y_val).item()
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
            row = {
                "epoch": epoch,
                "train_loss": round(total / len(train), 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
            }
            history.epochs.append(row)
            log(
                f"epoch {epoch:>2}  train_loss {row['train_loss']:.4f}  "
                f"val_loss {row['val_loss']:.4f}  val_acc {row['val_acc']:.4f}"
            )

            if val_loss < best_loss - 1e-4:
                best_loss, bad_epochs, history.best_epoch = val_loss, 0, epoch
                best_state = {k: v.detach().clone() for k, v in self.module.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    log(f"early stop: no val_loss improvement for {cfg.patience} epochs")
                    break

        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.module.eval()
        return history

    @torch.no_grad()
    def logits(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        self.module.eval()
        feats = self._prepare(texts)
        out = [self._forward(idx, feats).numpy() for idx in _batches(len(texts), batch_size, None)]
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)), dtype=np.float32)

    def _hidden(self, idx: np.ndarray, features) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 512) -> np.ndarray:
        self.module.eval()
        feats = self._prepare(texts)
        out = [self._hidden(idx, feats).numpy() for idx in _batches(len(texts), batch_size, None)]
        vecs = np.concatenate(out).astype(np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

    # Persistence -----------------------------------------------------------

    def _config(self) -> dict:
        raise NotImplementedError

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), d / "weights.pt")
        meta = {"kind": self.kind, "labels": self.labels.names, "config": self._config()}
        (d / "model.json").write_text(json.dumps(meta, indent=2))


class _FastTextModule(nn.Module):
    def __init__(self, n_buckets: int, dim: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        # sparse=True: each batch touches a few thousand of the 262k hashed rows, so
        # sparse gradients make a step O(rows used) instead of O(table size).
        self.embedding = nn.EmbeddingBag(n_buckets, dim, mode="mean", sparse=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, n_classes)
        nn.init.uniform_(self.embedding.weight, -1.0 / dim, 1.0 / dim)

    def hidden(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.embedding(ids, offsets)

    def forward(self, ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.head(self.dropout(self.hidden(ids, offsets)))


class FastTextClassifier(_TorchClassifier):
    kind = "fasttext"

    def __init__(
        self, labels: LabelSet, n_buckets: int = 2**18, dim: int = 128, dropout: float = 0.2
    ) -> None:
        self.labels = labels
        self.n_buckets, self.dim, self.dropout = n_buckets, dim, dropout
        self.module = _FastTextModule(n_buckets, dim, len(labels), dropout)

    def _optimizers(self, cfg: TrainConfig) -> list[torch.optim.Optimizer]:
        return [
            torch.optim.SparseAdam(list(self.module.embedding.parameters()), lr=cfg.lr),
            torch.optim.AdamW(
                self.module.head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            ),
        ]

    def _prepare(self, texts: list[str]) -> list[list[int]]:
        return [featurize(t, self.n_buckets) for t in texts]

    @staticmethod
    def _pack(idx: np.ndarray, feats: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = [feats[i] for i in idx]
        offsets = torch.tensor([0] + [len(r) for r in rows[:-1]]).cumsum(0)
        ids = torch.tensor([t for r in rows for t in r])
        return ids, offsets

    def _forward(self, idx, feats):
        return self.module(*self._pack(idx, feats))

    def _hidden(self, idx, feats):
        return self.module.hidden(*self._pack(idx, feats))

    def _config(self) -> dict:
        return {"n_buckets": self.n_buckets, "dim": self.dim, "dropout": self.dropout}


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU())
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


class EmbeddingMLPClassifier(_TorchClassifier):
    kind = "embedding-mlp"

    def __init__(
        self,
        labels: LabelSet,
        encoder_name: str = "BAAI/bge-small-en-v1.5",
        hidden: int = 512,
        dropout: float = 0.2,
        encoder=None,
    ) -> None:
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(encoder_name, device="cpu")
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.labels = labels
        self.hidden, self.dropout = hidden, dropout
        dim_fn = (
            getattr(encoder, "get_embedding_dimension", None)
            or encoder.get_sentence_embedding_dimension
        )
        in_dim = dim_fn()
        self.module = _MLPHead(in_dim, hidden, len(labels), dropout)
        self._cache: dict[str, np.ndarray] = {}

    def _prepare(self, texts: list[str]) -> torch.Tensor:
        # The encoder is frozen, so embeddings are computed once per text and reused
        # across epochs and between logits() / embed() calls.
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self.encoder.encode(missing, batch_size=64, normalize_embeddings=True)
            self._cache.update(zip(missing, np.asarray(vecs, dtype=np.float32)))
        return torch.from_numpy(np.stack([self._cache[t] for t in texts]))

    def _forward(self, idx, feats):
        return self.module(feats[torch.as_tensor(idx)])

    def _hidden(self, idx, feats):
        return feats[torch.as_tensor(idx)]

    def _config(self) -> dict:
        return {"encoder_name": self.encoder_name, "hidden": self.hidden, "dropout": self.dropout}


def build_model(kind: str, labels: LabelSet, **kwargs) -> _TorchClassifier:
    if kind == FastTextClassifier.kind:
        return FastTextClassifier(labels, **kwargs)
    if kind == EmbeddingMLPClassifier.kind:
        return EmbeddingMLPClassifier(labels, **kwargs)
    raise ValueError(f"unknown model kind {kind!r}; expected 'fasttext' or 'embedding-mlp'")


def load_model(directory: str | Path) -> _TorchClassifier:
    d = Path(directory)
    meta = json.loads((d / "model.json").read_text())
    model = build_model(meta["kind"], LabelSet(meta["labels"]), **meta["config"])
    model.module.load_state_dict(torch.load(d / "weights.pt", weights_only=True))
    model.module.eval()
    return model


def train_config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
