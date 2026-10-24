"""Lion optimizer (paper uses Lion over AdamW, Tab.5: +3.7pts).

MLX port: sign-momentum update; works on numpy arrays identically so we can
test logic before mlx install./detach
Update: m = b1*m + (1-b1)*g ; theta -= lr * sign(m + ... ) ; wd decoupled.
Simplified faithful version: m = beta1*m + (1-beta1)*g; theta -= lr*(sign(m)+wd*theta)
"""
from __future__ import annotations
import numpy as np


class Lion:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-4,
                 beta1: float = 0.9, beta2: float = 0.95, wd: float = 1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.wd = wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for k, v in params.items():
            g = grads[k]
            m = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.m[k] = self.beta2 * self.m[k] + (1 - self.beta2) * g
            upd = np.sign(m) + self.wd * v
            out[k] = (v - self.lr * upd).astype(v.dtype)
        return out
