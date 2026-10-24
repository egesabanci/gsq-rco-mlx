"""GSQ for MLX — Gumbel-Softmax scalar quantization (paper Sec 3).

Design: pure functions over (numpy | mlx) arrays via a tiny backend shim,
so smoke tests run without mlx installed and the same code runs on M3 with mlx.
"""
from __future__ import annotations
import math
import numpy as np

try:
    import mlx.core as mx  # type: ignore
    _HAS_MLX = True
except Exception:
    mx = None  # type: ignore
    _HAS_MLX = False


def get_grid(n_bits: int, kind: str = "uniform") -> np.ndarray:
    """Symmetric uniform grid, paper Eq.5. 2-bit -> [-2,-1,0,1]."""
    if n_bits == 1:  # ternary special-case handled by mask+sign, keep 3 levels
        return np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    k = 2 ** n_bits
    # paper uses G2 = {-2,-1,0,1} for 2-bit (skewed, scale sign absorbs bias)
    vals = np.arange(k, dtype=np.float32) - (k // 2)
    if n_bits == 2:
        vals = np.array([-2, -1, 0, 1], dtype=np.float32)
    return vals


def gptq_init_logits(w: np.ndarray, grid: np.ndarray, alpha: float = 6.0,
                     std: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    """Warm-start logits around nearest-grid (GPTQ proxy = RTN here).

    Real GPTQ needs Hessian; for tiny MLX runs RTN-nearest is the init and the
    GSQ optimization recovers most of the gap (paper App.A).
    Returns logits [d, K].
    """
    rng = rng or np.random.default_rng(0)
    d = w.shape[0]
    K = len(grid)
    # nearest grid index per weight (ignoring scale; scale fit separately)
    s0 = np.abs(w).mean() + 1e-8
    idx = np.argmin(np.abs(w[:, None] / s0 - grid[None, :]), axis=1)
    logits = -((np.arange(K)[None, :] - idx[:, None]) ** 2).astype(np.float32)
    noise = rng.standard_normal((d, K)).astype(np.float32)
    return std * (noise + alpha * logits / K)


def gumbel_softmax_sample(logits: np.ndarray, tau: float, kappa: float,
                          rng: np.random.Generator) -> np.ndarray:
    """Paper Alg.1: p = softmax((kappa*l + g)/tau), soft sample weights."""
    u = rng.random(logits.shape).astype(np.float64)
    g = -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))
    y = (kappa * logits + g) / max(tau, 1e-6)
    y = y - y.max(axis=1, keepdims=True)
    e = np.exp(y)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def quantize_soft(probs: np.ndarray, grid: np.ndarray, scale: float) -> np.ndarray:
    return (probs @ grid).astype(np.float32) * float(scale)


def quantize_hard(logits: np.ndarray, grid: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    idx = logits.argmax(axis=1)
    return (grid[idx] * float(scale)).astype(np.float32), idx


def local_shift_indices(j0: np.ndarray, shift_logits: np.ndarray, n_levels: int) -> np.ndarray:
    """Paper Sec 3.2, Fig.1: j = clip(j0 + delta), delta in {-2..2}."""
    delta = shift_logits.argmax(axis=1) - 2
    return np.clip(j0 + delta, 0, n_levels - 1)


def linear_schedule(t: int, T: int, start: float, end: float) -> float:
    return start + (end - start) * min(1.0, t / max(1, T - 1))
