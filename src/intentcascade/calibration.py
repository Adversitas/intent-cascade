"""Temperature scaling (Guo et al., 2017).

A single scalar T is fitted on held-out logits to minimise negative
log-likelihood; probabilities become softmax(logits / T). It doesn't change
which class wins, only how confident the model claims to be, and that is the
number the cascade thresholds on.
"""

from __future__ import annotations

import numpy as np
import torch


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TemperatureScaler:
    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = float(temperature)

    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 200) -> TemperatureScaler:
        x = torch.as_tensor(logits, dtype=torch.float64)
        y = torch.as_tensor(labels, dtype=torch.long)
        # Optimise log T so the temperature stays positive.
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(x / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        self.temperature = float(log_t.exp().item())
        return self

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, self.temperature)
