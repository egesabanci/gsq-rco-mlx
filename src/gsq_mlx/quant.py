"""GSQ for MLX — Gumbel-Softmax scalar quantization (paper Sec 3, Apps A/G).

Dual backend: every op accepts numpy arrays (fallback, keeps
``scripts/smoke_test.py`` green without mlx) AND ``mlx.core`` arrays
(real on-device path). Dispatch rule: if any input is an ``mx.array``,
the whole op runs in pure ``mx`` ops; otherwise numpy.

Grad rule (so losses are usable under ``mx.grad`` / ``mx.value_and_grad``):
sampling (randomness) lives in ``sample_gumbel`` / ``*_sample`` wrappers,
while the deterministic maps ``gs_softmax``, ``binary_gs_prob``,
``quantize_soft``, ``local_shift_values`` use only differentiable ops of
their backend — no RNG, no Python-side indexing inside the diff path.

Paper mapping:
  Alg.1  -> sample_gumbel + gs_softmax (+ gumbel_softmax_sample wrapper)
  Alg.2  -> ternary_soft / ternary_hard (binary mask + sign, Eq.2-4)
  Sec 3.2 2-bit  -> full-grid logits over G2 = {-2,-1,0,1} (Eq.5)
  Sec 3.2 b>2   -> local-shift deltas {-2..2}, j = clip(j0+d) (Eq.6, Fig.1)
  App A   -> init helpers live in gptq.py (Gaussian prior around GPTQ/RTN)
  App G   -> TAU 2 -> 0.05 linear, KAPPA 100 -> 500 linear,
             Lion lr 1e-4 (logits) / 5e-5 (scales), wd 1.0, betas (0.9, 0.95)
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


def _grid_as(probs, grid) -> object:
    """Grid as same-backend 1-D vector with probs dtype."""
    if _is_mx(probs):
        if _is_mx(grid):
            return grid.astype(probs.dtype)
        return mx.array(np.asarray(grid, dtype=np.float32)).astype(probs.dtype)
    return np.asarray(grid, dtype=np.float32).astype(probs.dtype)


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

def sample_gumbel(shape, rng: np.random.Generator | None = None,
                  key=None, dtype=np.float32):
    """Standard Gumbel(0,1) noise. Numpy if ``rng`` given (or no mlx),
    else ``mx.random.gumbel`` (optionally with explicit ``key``)."""
    if key is not None and _HAS_MLX:
        return mx.random.gumbel(shape=list(shape), key=key).astype(
            mx.float32 if dtype in (np.float32, float) else mx.float32)
    if rng is not None or not _HAS_MLX:
        r = rng or np.random.default_rng(0)
        u = r.random(tuple(shape)).astype(np.float64)
        g = -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))
        return g.astype(dtype)
    # mlx default stream (no key): still reproducible via mx.random.seed
    return mx.random.gumbel(shape=list(shape))


def gs_softmax(logits, g, tau: float, kappa: float):
    """Pure Alg.1 core: p = softmax((kappa*l + g)/tau). No RNG inside,
    so it is directly usable under ``mx.grad`` when inputs are mx arrays."""
    tau = max(float(tau), 1e-6)
    if _use_mx(logits, g):
        y = (float(kappa) * logits + g) / tau
        return mx.softmax(y, axis=-1)
    y = (float(kappa) * logits + np.asarray(g)) / tau
    y = y - y.max(axis=-1, keepdims=True)
    e = np.exp(y)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.result_type(y.dtype, np.float32))


def gumbel_softmax_sample(logits, tau: float, kappa: float,
                          rng: np.random.Generator | None = None,
                          key=None):
    """Paper Alg.1: p = softmax((kappa*l + g)/tau), g ~ Gumbel(0,1).

    ``rng`` (numpy Generator) selects the numpy path — this is the
    smoke_test calling convention; ``key`` selects the mlx path.
    An mx-array ``logits`` with no ``rng`` also takes the mlx path.
    """
    if key is not None and _HAS_MLX:
        g = mx.random.gumbel(shape=list(logits.shape), key=key)
        return gs_softmax(logits, g, tau, kappa)
    if _is_mx(logits) and rng is None:
        g = mx.random.gumbel(shape=list(logits.shape))
        return gs_softmax(logits, g, tau, kappa)
    r = rng or np.random.default_rng(0)
    g = sample_gumbel(logits.shape, rng=r)
    return gs_softmax(np.asarray(logits), g, tau, kappa)


def binary_gs_prob(logit, g0, g1, tau: float, kappa: float):
    """Pure binary GS (Alg.2 lines 4-5): single logit l over {0,1} with
    pair logits {-l, +l} -> P(active) = sigmoid((2*k*l + g1 - g0)/tau)."""
    tau = max(float(tau), 1e-6)
    if _use_mx(logit, g0, g1):
        z = (2.0 * float(kappa) * logit + (g1 - g0)) / tau
        return mx.sigmoid(z)
    z = (2.0 * float(kappa) * np.asarray(logit)
         + (np.asarray(g1) - np.asarray(g0))) / tau
    z = np.clip(z, -50.0, 50.0)  # avoid exp overflow; sigmoid saturates anyway
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def binary_gs_sample(logit, tau: float, kappa: float,
                     rng: np.random.Generator | None = None, key=None,
                     g_pair=None):
    """Sampled binary decision in [0,1]. Pass ``g_pair=(g0,g1)`` to reuse
    fixed noise (keeps the loss pure for ``mx.grad``)."""
    if g_pair is None:
        shape = getattr(logit, "shape", None) or np.shape(logit)
        if key is not None and _HAS_MLX:
            k1, k2 = mx.random.split(key)
            g_pair = (mx.random.gumbel(shape=list(shape), key=k1),
                      mx.random.gumbel(shape=list(shape), key=k2))
        elif _is_mx(logit) and rng is None:
            g_pair = (mx.random.gumbel(shape=list(shape)),
                      mx.random.gumbel(shape=list(shape)))
        else:
            r = rng or np.random.default_rng(0)
            g_pair = (sample_gumbel(shape, rng=r), sample_gumbel(shape, rng=r))
    g0, g1 = g_pair
    return binary_gs_prob(logit, g0, g1, tau, kappa)


# ------------------------------------------------- quantize (full grid) ---

def quantize_soft(probs, grid, scale: float):
    """Soft weights: w = s * (p @ grid). Pure matmul, grad-safe."""
    g = _grid_as(probs, grid)
    if _is_mx(probs):
        return (probs @ g) * float(scale)
    return (np.asarray(probs) @ np.asarray(g)).astype(np.float32) * float(scale)


def quantize_hard(logits, grid, scale: float):
    """Hard weights: idx = argmax l, w = s * grid[idx]."""
    if _use_mx(logits):
        lg = logits if _is_mx(logits) else mx.array(np.asarray(logits))
        idx = mx.argmax(lg, axis=-1)
        g = _grid_as(logits if _is_mx(logits) else lg, grid)
        w = mx.take(g, idx) * float(scale)
        return w, idx
    idx = np.asarray(logits).argmax(axis=-1)
    g = np.asarray(grid, dtype=np.float32)
    return (g[idx] * float(scale)).astype(np.float32), idx


# ---------------------------------------------------------------- ternary ---

def ternary_soft(mask_logits, sign_logits, scale: float,
                 tau: float, kappa: float,
                 rng: np.random.Generator | None = None, key=None,
                 noise=None):
    """Paper Alg.2 lines 3-7: m~ = GS({0,1},{-lm,+lm}), b~ = GS({-1,1},
    {-lb,+lb}), w_bar = s * m~ * b~. ``noise`` = (gm0,gm1,gs0,gs1) reuses
    fixed Gumbel draws so the map stays pure under ``mx.grad``."""
    if noise is None:
        shape = getattr(mask_logits, "shape", None) or np.shape(mask_logits)
        if key is not None and _HAS_MLX:
            ks = mx.random.split(key, 4)
            noise = tuple(mx.random.gumbel(shape=list(shape), key=k)
                          for k in ks)
        elif _use_mx(mask_logits, sign_logits):
            noise = tuple(mx.random.gumbel(shape=list(shape)) for _ in range(4))
        else:
            r = rng or np.random.default_rng(0)
            noise = tuple(sample_gumbel(shape, rng=r) for _ in range(4))
    gm0, gm1, gs0, gs1 = noise
    m = binary_gs_prob(mask_logits, gm0, gm1, tau, kappa)       # (0,1)
    b = 2.0 * binary_gs_prob(sign_logits, gs0, gs1, tau, kappa) - 1.0  # (-1,1)
    if _use_mx(m, b):
        return (m * b) * float(scale)
    return ((np.asarray(m) * np.asarray(b)) * float(scale)).astype(np.float32)


def ternary_hard(mask_logits, sign_logits, scale: float):
    """Alg.2 line 11: m^ = 1{lm >= 0}, b^ = 2*1{lb >= 0} - 1, w^ = s*m^*b^."""
    if _use_mx(mask_logits, sign_logits):
        m = (mask_logits >= 0).astype(mx.float32)
        b = (2.0 * (sign_logits >= 0).astype(mx.float32) - 1.0)
        return m * b * float(scale), (m, b)
    m = (np.asarray(mask_logits) >= 0).astype(np.float32)
    b = (2.0 * (np.asarray(sign_logits) >= 0).astype(np.float32) - 1.0)
    return ((m * b) * float(scale)).astype(np.float32), (m, b)


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


def local_shift_quantize_soft(shift_probs, j0, grid, scale: float):
    """Soft local-shift weights: w = s * sum_k p_k * grid[clip(j0+k-2)]."""
    C = local_shift_values(j0, grid)
    if _use_mx(shift_probs, C):
        sp = shift_probs if _is_mx(shift_probs) else mx.array(np.asarray(shift_probs))
        return (mx.sum(sp * C, axis=-1)) * float(scale)
    return (np.asarray(shift_probs) * np.asarray(C)).sum(axis=-1).astype(
        np.float32) * float(scale)


def local_shift_decode(j0, shift_logits, grid, scale: float):
    """Hard local-shift decode: argmax delta -> clipped index -> grid value."""
    K = len(grid)
    j = local_shift_indices(j0, shift_logits, K)
    if _is_mx(j):
        g = _grid_as(j, grid)
        return mx.take(g, j) * float(scale), j
    g = np.asarray(grid, dtype=np.float32)
    jj = np.asarray(j)
    return (g[jj] * float(scale)).astype(np.float32), jj
