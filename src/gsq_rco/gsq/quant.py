"""Grid, schedules and local-shift helpers of GSQ (paper Sec 3.2, Eq.5-6, App G).

Used by gsq.staged, gsq.gptq, gsq.export and rco.database:
  get_grid(bits)                 symmetric grid of Eq.5 (2-bit -> [-2,-1,0,1]; 1-bit -> ternary [-1,0,1])
  gs_schedules(t, T)             tau linear 2 -> 0.05 and kappa linear 100 -> 500 (App G)
  local_shift_indices / local_shift_values   b > 2 local-shift candidates (Sec 3.2, Eq.6)

The local-shift helpers accept numpy or mlx arrays. The Gumbel-Softmax maps of Alg.1/Alg.2
that the tests check the MLX code against are reference code in tests/oracles.py.
"""
from __future__ import annotations
import numpy as np

try:
    import mlx.core as mx  # type: ignore
    _HAS_MLX = True
except Exception:
    mx = None  # type: ignore
    _HAS_MLX = False

# Paper App G defaults (linear schedules over block training).
TAU_START, TAU_END = 2.0, 0.05
KAPPA_START, KAPPA_END = 100.0, 500.0
SHIFT_DELTAS = (-2, -1, 0, 1, 2)
N_SHIFTS = len(SHIFT_DELTAS)


def _is_mx(x) -> bool:
    return _HAS_MLX and isinstance(x, mx.array)


def _use_mx(*xs) -> bool:
    return any(_is_mx(x) for x in xs)


# ---------------------------------------------------------------- grids ---

def get_grid(n_bits: int, kind: str = "uniform") -> np.ndarray:
    """Symmetric uniform grid, paper Eq.5. 2-bit -> [-2,-1,0,1]."""
    if n_bits == 1:  # ternary special-case handled by mask+sign, keep 3 levels
        return np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    k = 2 ** n_bits
    vals = np.arange(k, dtype=np.float32) - (k // 2)
    if n_bits == 2:
        vals = np.array([-2, -1, 0, 1], dtype=np.float32)
    return vals


# -------------------------------------------------------------- schedules ---

def linear_schedule(t: int, T: int, start: float, end: float) -> float:
    return start + (end - start) * min(1.0, t / max(1, T - 1))


def gs_schedules(t: int, T: int,
                 tau: tuple[float, float] = (TAU_START, TAU_END),
                 kappa: tuple[float, float] = (KAPPA_START, KAPPA_END),
                 ) -> tuple[float, float]:
    """Paper App G: tau linear 2 -> 0.05, kappa linear 100 -> 500."""
    return (linear_schedule(t, T, tau[0], tau[1]),
            linear_schedule(t, T, kappa[0], kappa[1]))


# ---------------------------------------------------------------- sampling ---


# ------------------------------------------------- quantize (full grid) ---


# ---------------------------------------------------------------- ternary ---


# --------------------------------------------------------------- local shift ---

def local_shift_indices(j0, shift_logits, n_levels: int):
    """Paper Sec 3.2, Fig.1: j = clip(j0 + delta), delta in {-2..2}."""
    if _use_mx(j0, shift_logits):
        sl = shift_logits if _is_mx(shift_logits) else mx.array(np.asarray(shift_logits))
        jj = j0 if _is_mx(j0) else mx.array(np.asarray(j0))
        delta = mx.argmax(sl, axis=-1) - 2
        return mx.clip(jj + delta, 0, n_levels - 1)
    delta = np.asarray(shift_logits).argmax(axis=-1) - 2
    return np.clip(np.asarray(j0) + delta, 0, n_levels - 1)


def local_shift_values(j0, grid):
    """Candidate values table C[d,5]: C[i,k] = grid[clip(j0[i]+k-2)].
    Same-backend as inputs; pure indexing (one-hot matmul), grad-safe
    w.r.t. the probabilities that later weight these candidates."""
    K = len(grid)
    if _use_mx(j0, grid):
        assert _HAS_MLX
        jj = j0 if _is_mx(j0) else mx.array(np.asarray(j0, dtype=np.int32))
        g = (grid.astype(mx.float32) if _is_mx(grid)
             else mx.array(np.asarray(grid, dtype=np.float32)))
        # one-hot over grid levels without integer gather in the diff path
        kk = mx.arange(K)
        cols = []
        for k in range(N_SHIFTS):
            jk = mx.clip(jj + (k - 2), 0, K - 1)
            oh = (kk[None, :] == jk[:, None]).astype(g.dtype)
            cols.append(oh @ g)
        out = mx.stack(cols, axis=-1)
        return out
    jj = np.asarray(j0)
    g = np.asarray(grid, dtype=np.float32)
    jk = np.clip(jj[:, None] + (np.arange(N_SHIFTS) - 2)[None, :], 0, K - 1)
    return g[jk]


