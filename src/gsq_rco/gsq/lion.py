"""Lion update rule for the GSQ logits and group scales (Chen et al.; paper App G).

    u = sign(beta1 * m + (1 - beta1) * g)
    m <- beta2 * m + (1 - beta2) * g
    theta <- theta - lr * (u + wd * theta)

Paper App G learning rates (Tables 7-9): logits 1e-4 in every row; group scales 5e-5 for
the scalar 2/3-bit setting (Table 7, the default). betas (0.9, 0.95), wd 1.0.
scale_lr applies the bit-width rule of the stage optimiser: group scales above 3 bits are frozen.
lion_update is the only Lion implementation in the package; gsq.staged compiles it with
mx.compile. The numpy reference optimizer used for testing lives in tests/oracles.py.
"""
from __future__ import annotations

import mlx.core as mx

# Paper App G / Tables 7-9.
LR_LOGITS = 1e-4
LR_SCALES = 5e-5        # Table 7, scalar 2/3-bit (default for group scales)
WEIGHT_DECAY = 1.0
BETAS = (0.9, 0.95)
# Name of the group-scale rule below, recorded in the quantize fingerprint (shards made under another rule
# are not resumed).
SCALE_LR_RULE = "frozen_above_3_bits"


def scale_lr(lr_scales: float, bits: int) -> float:
    """Group-scale step of a linear with ``bits`` in the stage optimiser (WP5c rule, commit c6f55b1).

    Lion's sign update moves each group scale by ~lr per step in absolute terms, while the GPTQ scale of a
    b-bit group is max|w| / 2^(b-1): the scales shrink about 2x per bit (medians on the RCO shards, relative
    to 3 bits: 2-bit 2.06, 4-bit 0.60, 5-bit 0.27, 6-bit 0.15). Table 7's 5e-5 was set for the 2/3-bit grids.
    Real layer 3 of the RCO allocation on its quantized prefix (128 x 512, 200 steps, mlp stage), hard objective:
      lr 5e-5 at every width (HEAD)       2.47e-3 -> 9.82e-3 (3.97x); down_proj MSE 31x its GPTQ init
      lr * 2^-(bits-3) above 3 bits       2.44e-3 -> 4.15e-3 (1.70x), still rises
      frozen above 3 bits (this rule)     2.44e-3 -> 7.43e-4 (0.31x); qk 0.95x, vo 0.72x
    Table 9 (GGUF Q3/4/5_K rows) also sets the group-scale lr to 0. The rule freezes the group scales of
    4..8-bit linears in the stage optimiser; App H (scale_finetune) is not affected. bits <= 3 return
    lr_scales unchanged, so the 2/3-bit setting is identical to before the rule.
    """
    return 0.0 if bits > 3 else lr_scales


def lion_update(P: dict, M: dict, G: dict, lr: dict,
                wd: float = WEIGHT_DECAY, betas: tuple[float, float] = BETAS):
    """One Lion step on dicts of mx arrays keyed by tensor name: returns (new params, new momentum).

    Pure in its inputs, so mx.compile can fuse it. Each tensor is updated independently.
    """
    b1, b2 = betas
    newP, newM = {}, {}
    for k in P:
        g = G[k]
        newP[k] = P[k] - lr[k] * (mx.sign(b1 * M[k] + (1 - b1) * g) + wd * P[k])
        newM[k] = b2 * M[k] + (1 - b2) * g
    return newP, newM
