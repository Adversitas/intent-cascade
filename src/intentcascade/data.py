"""Dataset loading and splitting.

The bundled data is BANKING77 (Casanueva et al., 2020, CC-BY-4.0): 13,083 real
customer-support messages for an online bank, labelled with 77 fine-grained
intents. Many intents are near-duplicates ("card_arrival" vs
"card_delivery_estimate"), which is what makes it a useful testbed for knowing
when a cheap model should defer to a stronger one.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Example:
    text: str
    label: str


def load_csv(path: str | Path) -> list[Example]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"text", "category"} <= rows[0].keys():
        raise ValueError(f"{path}: expected 'text' and 'category' columns")
    return [Example(r["text"].strip(), r["category"].strip()) for r in rows if r["text"].strip()]


def stratified_split(
    examples: list[Example], val_fraction: float = 0.1, seed: int = 0
) -> tuple[list[Example], list[Example]]:
    """Hold out `val_fraction` of every label, so rare intents appear in both splits."""
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) > 1 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class LabelSet:
    """Stable mapping between intent names and class indices."""

    def __init__(self, labels: list[str]) -> None:
        self.names = sorted(set(labels))
        self.index = {name: i for i, name in enumerate(self.names)}

    def __len__(self) -> int:
        return len(self.names)

    def encode(self, labels: list[str]) -> list[int]:
        return [self.index[label] for label in labels]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.names[i] for i in ids]


def humanize(label: str) -> str:
    """'card_arrival' -> 'card arrival', for prompts and display."""
    return label.replace("_", " ")
