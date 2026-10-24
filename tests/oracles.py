"""Reference implementations used only by the tests.

Nothing in this file is part of the gsq_rco package. Each function was removed from the package
because production code does not call it. The tests compare the package's MLX code with these
independent numpy and MLX formulas of the paper equations (Alg.1-2, Sec 3.2, App A, Eq.6), and use
the Phase-1 synthetic helpers (synthetic_weights, init_layer, synthetic_calibration) as inputs.
"""
from __future__ import annotations

import numpy as np

try:
    import mlx.core as mx  # type: ignore
    _HAS_MLX = True
except Exception:
    mx = None  # type: ignore
    _HAS_MLX = False

from gsq_rco.gsq.gptq import (init_logits_2bit_from_codes, init_shift_logits,
                              init_ternary_logits, quantize_rtn)
from gsq_rco.gsq.lion import LR_LOGITS, LR_SCALES
from gsq_rco.gsq.quant import (get_grid, gs_schedules, local_shift_indices,
                               local_shift_values)


def _is_mx(x) -> bool:
    return _HAS_MLX and isinstance(x, mx.array)


def _use_mx(*xs) -> bool:
    return any(_is_mx(x) for x in xs)


def _grid_as(probs, grid) -> object:
    """Grid as same-backend 1-D vector with probs dtype."""
    if _is_mx(probs):
        if _is_mx(grid):
            return grid.astype(probs.dtype)
        return mx.array(np.asarray(grid, dtype=np.float32)).astype(probs.dtype)
    return np.asarray(grid, dtype=np.float32).astype(probs.dtype)


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
    numpy oracle calling convention; ``key`` selects the mlx path.
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


def synthetic_weights(shape, seed: int = 0, scale: float = 0.02):
    """Tiny stand-in weights ~ N(0, scale^2). NEVER a model checkpoint."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(tuple(shape)).astype(np.float32)
            * np.float32(scale))


def init_layer(n_bits: int, shape, group_size: int = 64,
               seed: int = 0, w: np.ndarray | None = None) -> dict:
    """One-call Phase-1 init for a synthetic layer: signed RTN codes/scales plus
    the matching GSQ logits flavor (ternary mask+sign / 2-bit full grid /
    3-bit+ local shift). ``w`` may be supplied; else synthetic."""
    if w is None:
        w = synthetic_weights(shape, seed=seed)
    w = np.asarray(w, dtype=np.float32)
    rtn = quantize_rtn(w, n_bits, group_size)
    grid = get_grid(n_bits)
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
    return out


def synthetic_kl(p, target, eps: float = 1e-9):
    """Mean KL(target || p) over groups — synthetic stand-in for the App-B
    KL objective (real version: calibration-token KL via mlx_lm, Phase 2).
    Pure + grad-safe in both backends."""
    if _is_mx(p) or _is_mx(target):
        pp = p if _is_mx(p) else mx.array(np.asarray(p))
        tt = target if _is_mx(target) else mx.array(np.asarray(target))
        return mx.mean(mx.sum(tt * (mx.log(tt + eps) - mx.log(pp + eps)), axis=1))
    pp = np.asarray(p, dtype=np.float64)
    tt = np.asarray(target, dtype=np.float64)
    return float((tt * (np.log(tt + eps) - np.log(pp + eps))).sum(axis=1).mean())


def assemble_weights_stub(z: np.ndarray, values_per_option: np.ndarray):
    """Synthetic weight assembly: z [N,K] one-hot @ values [K] -> [N].

    Stand-in for App-B assembly (per-group dequant values from GPTQ
    residuals at the DP-chosen bitwidth). Real mlx_lm wiring — gathering
    quantized layer weights at mixed precision — is Phase-2 work.
    """
    z = np.asarray(z, dtype=np.float64)
    v = np.asarray(values_per_option, dtype=np.float64).reshape(-1)
    if z.shape[1] != v.shape[0]:
        raise ValueError(f"z {z.shape} vs values {v.shape}")
    return z @ v


def synthetic_calibration(num_seqs: int, seq_len: int, hidden: int,
                          seed: int = 0) -> np.ndarray:
    """STUB (Phase 2: real tokenizer + FineWeb-Edu/OpenThoughts tokens).
    Returns synthetic 'token' activations X [M, hidden], M = num_seqs*seq_len.
    Callers keep M tiny (e.g. 32x8) so Phase-1 tests run in seconds."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((int(num_seqs) * int(seq_len), int(hidden)),
                              dtype=np.float32)


def linear_mse_np(X: np.ndarray, Wq: np.ndarray, W: np.ndarray) -> float:
    """||X @ Wq.T - X @ W.T||^2_F / M (W: [out, in], X: [M, in])."""
    R = (X @ Wq.T - X @ W.T)
    return float((R * R).mean())


def _expand_scales(scales, in_f: int, group_size: int) -> np.ndarray:
    """Per-(row, group) scales -> per-weight [out, in]; groups over consecutive inputs."""
    return np.repeat(np.asarray(scales, dtype=np.float64), group_size, axis=1)[:, :in_f]


def _group_sum(A: np.ndarray, group_size: int) -> np.ndarray:
    """Sum A [out, in] over each (row, group) -> [out, n_groups]."""
    out_f, in_f = A.shape
    n_groups = -(-in_f // group_size)
    Ap = np.zeros((out_f, n_groups * group_size))
    Ap[:, :in_f] = A
    return Ap.reshape(out_f, n_groups, group_size).sum(axis=-1)


def hard_loss_np(X, W, logits, scales, grid, group_size: int) -> float:
    """Reconstruction loss of the hard weights: argmax codes and learned scales."""
    W = np.asarray(W, dtype=np.float64)
    idx = np.argmax(np.asarray(logits), axis=1)
    codes_val = np.asarray(grid, dtype=np.float64)[idx].reshape(W.shape)
    return linear_mse_np(X, codes_val * _expand_scales(scales, W.shape[1], group_size), W)


def linear_loss_and_grads_np(X: np.ndarray, W: np.ndarray, logits: np.ndarray,
                             grid: np.ndarray, scales: np.ndarray, group_size: int,
                             tau: float, kappa: float, n_resamples: int = 4,
                             seed: int = 0):
    """Loss ||X (w_bar - W)^T||^2 / (M * out), averaged over ``n_resamples`` Gumbel
    draws (paper: accumulation damps sampling variance). Returns
    (loss, dL/dlogits [d, K], dL/dscales [out, n_groups]).

    With w_bar = s * c, c = p @ grid and p = softmax((kappa*l + g)/tau):
      u_k = dL/dp_k = (s * dL/dw_bar) * grid_k,
      dL/dl = (kappa/tau) * p * (u - <p, u>),   dL/ds = sum over group of dL/dw_bar * c.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    out_f, in_f = W.shape
    logits = np.asarray(logits, dtype=np.float64)
    d, K = logits.shape
    assert d == W.size, "logits rows must match W elements"
    gridv = np.asarray(grid, dtype=np.float64)
    s_el = _expand_scales(scales, in_f, group_size)
    M = X.shape[0]
    acc_loss, acc_glog, acc_gs = 0.0, 0.0, 0.0
    for _ in range(n_resamples):
        g = sample_gumbel((d, K), rng=rng).astype(np.float64)
        p = gs_softmax(logits, g, tau, kappa)                      # [d, K]
        c = (p @ gridv).reshape(W.shape)
        R = X @ (s_el * c - W).T                                   # [M, out]
        acc_loss += float((R * R).mean())
        G = (2.0 / (M * out_f)) * (R.T @ X)                        # dL/dw_bar
        u = (s_el * G).reshape(-1)[:, None] * gridv[None, :]       # dL/dp
        glog = (kappa / max(tau, 1e-6)) * p * (u - (p * u).sum(axis=1, keepdims=True))
        acc_glog = acc_glog + glog
        acc_gs = acc_gs + _group_sum(G * c, group_size)
    return (acc_loss / n_resamples,
            acc_glog / n_resamples,
            acc_gs / n_resamples)


def train_linear_synthetic(n_bits: int = 2, in_f: int = 32, out_f: int = 16,
                           group_size: int = 16, steps: int = 1280,
                           n_resamples: int = 4, seed: int = 0,
                           lr_logits: float = LR_LOGITS,
                           lr_scales: float = LR_SCALES) -> dict:
    """Phase-1 synthetic block loop: GSQ-optimize (logits, group scales) of one linear.

    Init: App A logits around the signed-RTN codes (Eq.4 noise) and the signed-RTN
    scales. The tau/kappa schedule spans the whole run (paper App G). Returns the
    soft-loss history and the HARD reconstruction loss (argmax codes and learned
    scales) before and after training, the number of flipped codes, and the largest
    logit move. Only the 2-bit full grid is implemented; b>2 uses local-shift logits
    (Sec 3.2) and is not wired into this loop.
    """
    if n_bits != 2:
        raise NotImplementedError("b>2 uses local-shift logits (Sec 3.2); not wired here")
    grid = get_grid(n_bits)
    W = synthetic_weights((out_f, in_f), seed=seed + 1)
    X = synthetic_calibration(8, 16, in_f, seed=seed + 2)
    rtn = quantize_rtn(W, n_bits, group_size)
    logits = init_logits_2bit_from_codes(rtn["codes"], len(grid),
                                         seed=seed + 3).astype(np.float64)
    logits0 = logits.copy()
    scales = rtn["scales"].astype(np.float64)
    hard_init = hard_loss_np(X, W, logits, scales, grid, group_size)
    opt_logits = Lion({"logits": logits}, lr=lr_logits)
    opt_scales = Lion({"scales": scales}, lr=lr_scales)
    soft_history = []
    for t in range(steps):
        tau, kappa = gs_schedules(t, steps)
        loss, glog, gscale = linear_loss_and_grads_np(
            X, W, logits, grid, scales, group_size, tau, kappa,
            n_resamples=n_resamples, seed=seed + 1000 + t)
        soft_history.append(loss)
        logits = opt_logits.step({"logits": logits}, {"logits": glog})["logits"]
        scales = opt_scales.step({"scales": scales}, {"scales": gscale})["scales"]
    hard_final = hard_loss_np(X, W, logits, scales, grid, group_size)
    flips = int(np.sum(np.argmax(logits, axis=1) != np.argmax(logits0, axis=1)))
    return {"soft_history": soft_history, "hard_init": hard_init,
            "hard_final": hard_final, "n_flips": flips,
            "max_logit_move": float(np.abs(logits - logits0).max()),
            "params": {"logits": logits, "scales": scales},
            "grid": grid, "W": W, "X": X, "group_size": group_size}


def linear_soft_loss_mx(logits, scale, g, X, Wflat, grid, tau: float,
                        kappa: float):
    """Pure MLX staged linear loss: mean((X@(wq-wo))^2), wq from GS probs.
    ``g`` (Gumbel noise) is an explicit arg, so a closure over
    (logits, scale) with fixed (g, X, Wflat, grid) is usable under
    ``mx.value_and_grad(..., argnums=(0, 1))``. Weight matrices are flat
    [d]; the output dim is recovered from X's input dim. ``scale`` is a scalar
    or a flat per-weight [d] array (expand per-group scales with np.repeat)."""
    p = gs_softmax(logits, g, tau, kappa)
    gv = grid.astype(p.dtype) if hasattr(grid, "dtype") else grid
    c = p @ gv
    in_f = X.shape[-1]
    Wq = (scale * c).reshape(-1, in_f)
    W = Wflat.reshape(-1, in_f)
    R = X @ (Wq - W).T
    return mx.mean(R * R)
