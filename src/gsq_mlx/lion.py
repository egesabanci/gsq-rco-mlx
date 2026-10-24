"""Lion optimizer (paper uses Lion over AdamW, Tab.5: +3.7pts).

Update (Chen et al.; paper App G betas (0.9, 0.95), wd 1.0):
    u = sign(beta1 * m + (1 - beta1) * g)
    m <- beta2 * m + (1 - beta2) * g
    theta <- theta - lr * (u + wd * theta)

Paper App G learning rates (Tables 7-9): logits 1e-4 in every row. Group scales:
5e-5 for the scalar 2/3-bit setting (Table 7, LR_SCALES, the default); 2e-5 for
the GGUF Q2_K row of Table 9 (LR_SCALES_QWEN, kept for e2e.py until it switches).
scale_lr applies the bit-width rule of the stage optimiser: group scales above 3 bits are frozen.
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


def scale_lr(lr_scales: float, bits: int) -> float:
    """Group-scale step of a linear with ``bits`` in the stage optimiser (WP5c rule).

    Lion's sign update moves each group scale by ~lr per step in absolute terms, while the GPTQ
    scale of a b-bit group is max|w| / 2^(b-1) (gptq._group_scales): the scales shrink about 2x per
    bit (medians on the RCO shards, relative to 3 bits: 2-bit 2.06, 4-bit 0.60, 5-bit 0.27, 6-bit
    0.15). Table 7's 5e-5 was set for the 2/3-bit grids. Real layer 3 of the RCO allocation on its
    quantized prefix, 128 x 512 calibration, 200 steps, mlp stage (gate 4, up 3, down 5 bits), hard
    objective:
      lr 5e-5 at every width (HEAD)       2.47e-3 -> 9.82e-3 (3.97x); vo 6.92e-5 -> 8.18e-5;
                                          down_proj MSE 31x its GPTQ init
      lr * 2^-(bits-3) above 3 bits       2.44e-3 -> 4.15e-3 (1.70x), still rises
      frozen above 3 bits (this rule)     2.44e-3 -> 7.43e-4 (0.31x); qk 0.95x, vo 0.72x
    Table 9 (GGUF Q3/4/5_K rows) also sets the group-scale lr to 0. The rule freezes the group
    scales of 4..8-bit linears in the stage optimiser; App H (scale_finetune) is not affected.
    bits <= 3 return lr_scales unchanged, so the 2/3-bit setting and the uniform-3 run are bitwise
    identical to HEAD.
    """
    return 0.0 if bits > 3 else lr_scales


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
