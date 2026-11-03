"""A minimal file-based model registry.

Each training run writes a versioned bundle::

    models/
      fasttext-20261103-142501/
        model.json  weights.pt     the network (see models.py)
        calibration.json           fitted temperature
        metrics.json               validation + test metrics at training time
        card.json                  data, config and training history
      LATEST                       name of the bundle the API serves by default

The metrics stored at training time double as the reference distribution for
drift monitoring: production confidence is compared against the confidence
histogram recorded here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from .calibration import TemperatureScaler
from .models import _TorchClassifier, load_model

DEFAULT_REGISTRY = Path("models")


@dataclass
class Bundle:
    name: str
    path: Path
    model: _TorchClassifier
    scaler: TemperatureScaler
    metrics: dict
    card: dict


def new_version_name(kind: str) -> str:
    return f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}"


def save_bundle(
    registry: str | Path,
    name: str,
    model: _TorchClassifier,
    scaler: TemperatureScaler,
    metrics: dict,
    card: dict,
    set_latest: bool = True,
) -> Path:
    path = Path(registry) / name
    model.save(path)
    (path / "calibration.json").write_text(json.dumps({"temperature": scaler.temperature}))
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (path / "card.json").write_text(json.dumps(card, indent=2))
    if set_latest:
        (Path(registry) / "LATEST").write_text(name)
    return path


def resolve(registry: str | Path, name: str | None = None) -> Path:
    registry = Path(registry)
    if name is None:
        latest = registry / "LATEST"
        if not latest.exists():
            raise FileNotFoundError(
                f"no trained model in {registry}/ - run `intent-cascade train` first"
            )
        name = latest.read_text().strip()
    path = registry / name
    if not (path / "model.json").exists():
        raise FileNotFoundError(f"{path} is not a model bundle")
    return path


def load_bundle(registry: str | Path = DEFAULT_REGISTRY, name: str | None = None) -> Bundle:
    path = resolve(registry, name)
    temperature = json.loads((path / "calibration.json").read_text())["temperature"]
    return Bundle(
        name=path.name,
        path=path,
        model=load_model(path),
        scaler=TemperatureScaler(temperature),
        metrics=json.loads((path / "metrics.json").read_text()),
        card=json.loads((path / "card.json").read_text()),
    )


def list_bundles(registry: str | Path = DEFAULT_REGISTRY) -> list[dict]:
    registry = Path(registry)
    latest = (registry / "LATEST").read_text().strip() if (registry / "LATEST").exists() else None
    out = []
    for d in sorted(registry.glob("*/metrics.json")):
        m = json.loads(d.read_text()).get("test", {})
        out.append(
            {
                "name": d.parent.name,
                "latest": d.parent.name == latest,
                "test_accuracy": m.get("accuracy"),
                "test_ece": m.get("ece"),
            }
        )
    return out
