from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from intentcascade.calibration import TemperatureScaler
from intentcascade.data import DATA_DIR, Example, LabelSet, load_csv, stratified_split
from intentcascade.models import FastTextClassifier, TrainConfig

# A small slice of BANKING77 keeps the suite fast while still exercising real text.
SMOKE_LABELS = [
    "card_arrival",
    "lost_or_stolen_card",
    "exchange_rate",
    "top_up_failed",
    "cancel_transfer",
    "pin_blocked",
]


@pytest.fixture(scope="session")
def small_data() -> tuple[list[Example], list[Example], LabelSet]:
    train = [e for e in load_csv(DATA_DIR / "train.csv") if e.label in SMOKE_LABELS]
    test = [e for e in load_csv(DATA_DIR / "test.csv") if e.label in SMOKE_LABELS]
    return train, test, LabelSet([e.label for e in train])


@pytest.fixture(scope="session")
def trained(small_data):
    train, _, labels = small_data
    tr, val = stratified_split(train, 0.15, seed=0)
    model = FastTextClassifier(labels, n_buckets=2**14, dim=32)
    model.fit(tr, val, TrainConfig(epochs=25, lr=1e-2, patience=5), log=lambda *_: None)
    import numpy as np

    scaler = TemperatureScaler().fit(
        model.logits([e.text for e in val]), np.array(labels.encode([e.label for e in val]))
    )
    return model, scaler


class FakeMessages:
    """Records requests and replays a canned structured-output response."""

    def __init__(self, payload: dict | None = None, stop_reason: str = "end_turn") -> None:
        self.payload = payload or {}
        self.stop_reason = stop_reason
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            model=kwargs["model"],
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
            usage=SimpleNamespace(input_tokens=500, output_tokens=12),
        )


def fake_client(messages: FakeMessages) -> SimpleNamespace:
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=messages))
