from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class DualController:
    beta: float = 2.5
    dual_lr: float = 1e-4
    lambda_init: float = 1.0
    lambda_max: Optional[float] = None

    lam: float = 1.0

    def __post_init__(self):
        self.lam = float(self.lambda_init)

    def step(self, f_mean: float) -> float:
        # \lambda_{k+1} = \max\{0, \lambda_k + \alpha (f(x_{k+1})-\beta)\}
        g = float(f_mean - self.beta)
        self.lam = max(0.0, float(self.lam + self.dual_lr * g))

        if self.lambda_max is not None:
            self.lam = min(self.lam, float(self.lambda_max))
        return self.lam
