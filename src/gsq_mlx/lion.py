"""Lion optimizer (paper uses Lion over AdamW, Tab.5: +3.7pts).

Update (Chen et al.; paper App G betas (0.9, 0.95), wd 1.0):
    u = sign(beta1 * m + (1 - beta1) * g)
    m <- beta2 * m + (1 - beta2) * g
    theta <- theta - lr * (u + wd * theta)

Paper App G learning rates (Tables 7-9): logits 1e-4 in every row. Group scales:
5e-5 for the scalar 2/3-bit setting (Table 7, LR_SCALES, the default); 2e-5 for
the GGUF Q2_K row of Table 9 (LR_SCALES_QWEN, kept for e2e.py until it switches).
Dual backend: numpy dicts (tests) and mx arrays (device training) via the same call.
"""
from __future__ import annotations
import numpy as np

try:
    import mlx.core as mx  # type: ignore
    _HAS_MLX = True
except Exception:
    mx = None  # type: ignore
    _HAS_MLX = False

# Paper App G / Tables 7-9.
LR_LOGITS = 1e-4
LR_SCALES = 5e-5        # Table 7, scalar 2/3-bit (default for group scales)
LR_SCALES_QWEN = 2e-5   # Table 9, Q2_K (GGUF K-quant) row; backward compatibility
WEIGHT_DECAY = 1.0
BETAS = (0.9, 0.95)


def _is_mx(x) -> bool:
    return _HAS_MLX and isinstance(x, mx.array)


class Lion:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-4,
                 beta1: float = 0.9, beta2: float = 0.95, wd: float = 1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.wd = wd
        self.m = {}
        for k, v in params.items():
            if _is_mx(v):
                self.m[k] = mx.zeros_like(v)
            else:
                self.m[k] = np.zeros_like(v)

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for k, v in params.items():
            g = grads[k]
            if _is_mx(v) or _is_mx(g) or _is_mx(self.m[k]):
                m = self.beta1 * self.m[k] + (1 - self.beta1) * g
                self.m[k] = self.beta2 * self.m[k] + (1 - self.beta2) * g
                upd = mx.sign(m) + self.wd * v
                out[k] = (v - self.lr * upd).astype(v.dtype)
            else:
                m = self.beta1 * self.m[k] + (1 - self.beta1) * g
                self.m[k] = self.beta2 * self.m[k] + (1 - self.beta2) * g
                upd = np.sign(m) + self.wd * np.asarray(v)
                out[k] = (np.asarray(v) - self.lr * upd).astype(np.asarray(v).dtype)
        return out
