"""GPTQ-lite / RTN init for GSQ (paper Sec 3.1-3.2, App A).

Phase 1 scope (MEMORY CRITICAL — Mac 17GB, ~2.5GB free):
  * NEVER loads models/*.safetensors (942MB). Only ``config.json`` shapes
    plus synthetic weights are used.
  * Full GPTQ needs activation Hessians from calibration data; Phase 1 uses
    the RTN-nearest proxy (row-wise, per-group), which is exactly what the
    paper's App-A init reduces to before GSQ optimization takes over.
    Hessian-aware init is Phase-2 work (real calibration activations).

Group convention (paper Tab 7-9: group_size 128; Phase 1 uses 64 to stay
tiny-friendly): groups are row-wise over consecutive entries of the last
axis; each group gets one symmetric scale s = max|w_g| / max|grid|.

Provides:
  load_qwen_shapes   config.json -> {name: (out, in)} linear shapes
  rtn_quantize_groupwise   (scales, codes j0, w_q) for any 2-D weight
  init_logits_2bit_from_codes / init_shift_logits / init_ternary_logits
  init_layer  one-call init for a (synthetic) layer, 1/2/3+-bit aware
"""
from __future__ import annotations
import json
import numpy as np


def load_qwen_shapes(config_path: str) -> dict[str, tuple[int, int]]:
    """Read ONLY config.json (never safetensors) and derive Qwen2 linear
    shapes as (out_features, in_features). Returns per-block linears."""
    with open(config_path) as f:
        cfg = json.load(f)
    h = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    n_h = int(cfg["num_attention_heads"])
    n_kv = int(cfg.get("num_key_value_heads", n_h))
    hd = h // n_h
    kv = n_kv * hd
    return {
        "q_proj": (h, h),
        "k_proj": (kv, h),
        "v_proj": (kv, h),
        "o_proj": (h, h),
        "gate_proj": (inter, h),
        "up_proj": (inter, h),
        "down_proj": (h, inter),
    }


def synthetic_weights(shape, seed: int = 0, scale: float = 0.02):
    """Tiny stand-in weights ~ N(0, scale^2). NEVER a model checkpoint."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(tuple(shape)).astype(np.float32)
            * np.float32(scale))


def rtn_quantize_groupwise(w: np.ndarray, grid: np.ndarray,
                           group_size: int = 64):
    """RTN-nearest per-group symmetric quant (GPTQ-lite init).

    Returns dict(scales [n_groups], codes j0 [same shape as w], w_q).
    Groups tile the last axis; a ragged tail forms its own group.
    """
    w = np.asarray(w, dtype=np.float32)
    g = np.asarray(grid, dtype=np.float32)
    gmax = float(np.abs(g).max())
    flat = w.reshape(-1, w.shape[-1])
    n_groups = int(np.ceil(flat.shape[1] / group_size))
    scales = np.zeros((flat.shape[0], n_groups), dtype=np.float32)
    codes = np.zeros_like(flat, dtype=np.int64)
    wq = np.zeros_like(flat)
    for r in range(flat.shape[0]):
        for gi in range(n_groups):
            sl = slice(gi * group_size, min((gi + 1) * group_size, flat.shape[1]))
            grp = flat[r, sl]
            s = float(np.abs(grp).max() / gmax) if gmax > 0 else 0.0
            s = max(s, 1e-8)
            scales[r, gi] = s
            j = np.argmin(np.abs(grp[:, None] / s - g[None, :]), axis=1)
            codes[r, sl] = j
            wq[r, sl] = g[j] * s
    shape = w.shape
    return {"scales": scales, "codes": codes.reshape(shape),
            "w_q": wq.reshape(shape)}


def _noisy_alpha_init(base: np.ndarray, alpha: float, std: float,
                      seed: int) -> np.ndarray:
    """App A / Eq.4: l = std * (eps + alpha * l_gptq), eps ~ N(0,I),
    with l_gptq mean-subtracted per coordinate."""
    rng = np.random.default_rng(seed)
    b = base - base.mean(axis=-1, keepdims=True)
    eps = rng.standard_normal(base.shape).astype(np.float32)
    return (std * (eps + alpha * b)).astype(np.float32)


def init_logits_2bit_from_codes(codes: np.ndarray, K: int,
                               alpha: float = 6.0, std: float = 0.01,
                               seed: int = 0) -> np.ndarray:
    """App A (b=2): candidates are grid points, mu_i = GPTQ value, so
    (l_gptq)_{i,k} ~ -(k - q_i)^2 / 2, then Eq.4 noise."""
    codes = np.asarray(codes)
    kk = np.arange(K, dtype=np.float32)
    base = -((kk[None, :] - codes.reshape(-1)[:, None].astype(np.float32)) ** 2) / 2.0
    return _noisy_alpha_init(base.astype(np.float32), alpha, std, seed)


def init_shift_logits(numel: int, alpha: float = 6.0, std: float = 0.01,
                      seed: int = 0) -> np.ndarray:
    """App A (b>2): candidates are shifts {-2..2}, GPTQ solution already
    encoded in j0, so mu = 0 (stay at GPTQ point): base ~ -(c_k)^2 / 2."""
    c = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    base = np.tile(-(c ** 2) / 2.0, (int(numel), 1))
    return _noisy_alpha_init(base, alpha, std, seed)


def init_ternary_logits(q_gptq: np.ndarray, alpha: float = 3.0,
                        std: float = 0.01, seed: int = 0):
    """Eq.3-4: mask logit +1 active / -1 pruned; sign logit +1/-1/0."""
    q = np.asarray(q_gptq).reshape(-1)
    lm = np.where(q != 0, 1.0, -1.0).astype(np.float32)
    lb = np.where(q > 0, 1.0, np.where(q < 0, -1.0, 0.0)).astype(np.float32)
    rng = np.random.default_rng(seed)
    eps_m = rng.standard_normal(q.shape).astype(np.float32)
    eps_b = rng.standard_normal(q.shape).astype(np.float32)
    return (std * (eps_m + alpha * lm)).astype(np.float32), \
           (std * (eps_b + alpha * lb)).astype(np.float32)


def init_layer(n_bits: int, shape, group_size: int = 64,
               seed: int = 0, w: np.ndarray | None = None) -> dict:
    """One-call Phase-1 init for a synthetic layer: RTN codes/scales plus
    the matching GSQ logits flavor (ternary mask+sign / 2-bit full grid /
    3-bit+ local shift). ``w`` may be supplied; else synthetic."""
    from .quant import get_grid
    grid = get_grid(n_bits)
    if w is None:
        w = synthetic_weights(shape, seed=seed)
    w = np.asarray(w, dtype=np.float32)
    rtn = rtn_quantize_groupwise(w, grid, group_size)
    out = {"grid": grid, "scales": rtn["scales"], "j0": rtn["codes"],
           "w_q": rtn["w_q"], "group_size": group_size}
    if n_bits == 1:
        lm, lb = init_ternary_logits(np.sign(rtn["w_q"]), seed=seed)
        out.update({"mask_logits": lm, "sign_logits": lb})
    elif n_bits == 2:
        out["logits"] = init_logits_2bit_from_codes(
            rtn["codes"], len(grid), seed=seed)
    else:
        out["shift_logits"] = init_shift_logits(w.size, seed=seed)
        out["logits_full"] = None  # naive 2^b path deliberately not built
    return out
