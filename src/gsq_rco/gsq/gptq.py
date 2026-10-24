"""Quantizers and logit inits for GSQ (paper Sec 3.1-3.2, App A).

Shared contract (names and shapes are fixed; gsq.pipeline, gsq.export, rco.database and the tests code against it):
  quantize_rtn(W, bits, group_size, signed_scale)  -> {codes u8 [out,in], scales f32 [out,G], w_q f32 [out,in]}
  quantize_gptq(W, H, bits, ...)                   -> same dict + "err"
  hessian_from_inputs(X, H, n)                     -> (H, n), H = 2 X^T X / M in float64
  dequantize(codes, scales, bits, group_size)      -> f32 [out,in]
with G = ceil(in/group_size), codes indexing quant.get_grid(bits), and
w_q[:, c] = scales[:, c // group_size] * grid[codes[:, c]].

Scales (Sec 3.2): |s| = max|w_group| / max|grid|. The grid is skewed toward negative values,
so the sign of s is chosen per group to minimise the group's squared error ("we allow the
scale s to take negative values as well"). Groups are row-wise over consecutive inputs.

This module never reads safetensors: weights and calibration activations come from callers.
"""
from __future__ import annotations
import numpy as np

from .quant import get_grid


# ------------------------------------------------------------ RTN and GPTQ ---

def _grid64(bits: int) -> np.ndarray:
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8 (uint8 codes), got {bits}")
    return get_grid(bits).astype(np.float64)


def _encode(X, s, grid):
    """Eq.5 codes: index of the grid value nearest X/s, clipped to the grid.
    A zero scale encodes to the zero value, whose dequantized weight is 0."""
    s_safe = np.where(s == 0, 1.0, s)
    q = np.clip(np.rint(X / s_safe), grid[0], grid[-1])
    return (q - grid[0]).astype(np.uint8)


def _group_sq_err(Wg, s, grid):
    return ((Wg - s[..., None] * grid[_encode(Wg, s[..., None], grid)]) ** 2).sum(axis=-1)


def _group_scales(Wg, grid, signed):
    """Scales for grouped weights Wg [out, n_groups, gs] -> f32 [out, n_groups].

    |s| = max|w| / max|grid|. With ``signed`` the sign minimises the group's squared error,
    so the largest-magnitude weight can use the long side of the asymmetric grid (Sec 3.2)."""
    s = (np.abs(Wg).max(axis=-1) / np.abs(grid).max()).astype(np.float32).astype(np.float64)
    if signed:
        s = np.where(_group_sq_err(Wg, -s, grid) < _group_sq_err(Wg, s, grid), -s, s)
    return s.astype(np.float32)


def _to_groups(W, group_size):
    """Zero-pad the ragged last group; zeros encode exactly and add no error."""
    out_f, in_f = W.shape
    n_groups = -(-in_f // group_size)
    Wp = np.zeros((out_f, n_groups * group_size))
    Wp[:, :in_f] = W
    return Wp.reshape(out_f, n_groups, group_size)


def dequantize(codes, scales, bits, group_size=64):
    """w_q[:, c] = scales[:, c // group_size] * grid[codes[:, c]] (float32)."""
    grid = get_grid(bits)
    codes = np.asarray(codes)
    out_f, in_f = codes.shape
    scales = np.asarray(scales, dtype=np.float32)
    n_groups = -(-in_f // group_size)
    if scales.shape != (out_f, n_groups):
        raise ValueError(f"scales must be [{out_f}, {n_groups}], got {scales.shape}")
    gid = np.arange(in_f) // group_size
    return (scales[:, gid] * grid[codes]).astype(np.float32)


def quantize_rtn(W, bits, group_size=64, signed_scale=True):
    """Round-to-nearest on the grid with per-(row, group) signed scales (Sec 3.2)."""
    grid = _grid64(bits)
    W = np.asarray(W, dtype=np.float64)
    out_f, in_f = W.shape
    Wg = _to_groups(W, group_size)
    scales = _group_scales(Wg, grid, signed_scale)
    codes = _encode(Wg, scales.astype(np.float64)[..., None], grid)
    codes = np.ascontiguousarray(codes.reshape(out_f, -1)[:, :in_f])
    return {"codes": codes, "scales": scales,
            "w_q": dequantize(codes, scales, bits, group_size)}


def hessian_from_inputs(X, H=None, n=0):
    """Streaming input Hessian H = 2 X^T X / M in float64. Start with (H, n) = (None, 0)
    and feed the returned (H, n) back with each batch X: [m, in]. The accumulator is
    updated in place and returned with the new sample count n + m."""
    X = np.asarray(X, dtype=np.float64)
    total = n + X.shape[0]
    if H is None:
        H = np.zeros((X.shape[1], X.shape[1]))
    H *= n / total
    G = X.T @ X
    G *= 2.0 / total
    H += G
    return H, total


def _tri_inv_lower(L: np.ndarray, block: int = 512) -> np.ndarray:
    """Inverse of a lower-triangular matrix by blocked forward substitution.

    Holds one n x n output buffer. A dense inv holds its input, an identity and an
    output copy at once (about 0.57 GB extra at n = 4864 in float64)."""
    n = L.shape[0]
    X = np.zeros_like(L)
    for i in range(0, n, block):
        ie = min(i + block, n)
        Dinv = np.linalg.inv(L[i:ie, i:ie])
        X[i:ie, i:ie] = Dinv
        for j in range(0, i, block):  # block starts are aligned, so j + block <= i
            acc = np.zeros((ie - i, block))
            for k in range(j, i, block):
                acc += L[i:ie, k:k + block] @ X[k:k + block, j:j + block]
            X[i:ie, j:j + block] = -Dinv @ acc
    return X


def quantize_gptq(W, H, bits, group_size=64, damp=0.01, block_size=128,
                  act_order=False, signed_scale=True):
    """GPTQ (Frantar et al. 2022) on the same grid, scales and codes as quantize_rtn.

    Minimises ||X (W_q - W)^T||^2 for H = 2 X^T X / M. Columns are quantized left to right; each
    column's error is pushed onto the remaining columns through the upper Cholesky factor U of
    (H + damp * mean(diag H) I)^-1, applied lazily once per ``block_size`` block. A group's scales
    are set when its first column is reached, from the error-updated weights, with the RTN signed
    rule. Dead inputs (H_ii = 0) get a unit diagonal and a zero weight column.

    ``err`` = 1/2 tr(dW H dW^T) with the undamped H, i.e. the mean squared calibration output error.
    """
    if act_order:
        raise NotImplementedError("act_order needs a permuted static-group layout; not implemented")
    grid = _grid64(bits)
    W0 = np.asarray(W, dtype=np.float64)
    H0 = np.asarray(H, dtype=np.float64)
    out_f, in_f = W0.shape
    if H0.shape != (in_f, in_f):
        raise ValueError(f"H must be [{in_f}, {in_f}], got {H0.shape}")
    W = W0.copy()
    Hd = H0.copy()
    dead = np.flatnonzero(np.diag(Hd) == 0)
    Hd[dead, dead] = 1.0
    W[:, dead] = 0.0
    diag = np.arange(in_f)
    Hd[diag, diag] += damp * Hd[diag, diag].mean()
    L = np.linalg.cholesky(Hd)      # Hd = L L^T
    del Hd
    Linv = _tri_inv_lower(L)        # Hd^-1 = Linv^T Linv
    del L
    Hinv = Linv.T @ Linv
    del Linv
    U = np.linalg.cholesky(Hinv).T  # upper, Hinv = U^T U
    del Hinv

    n_groups = -(-in_f // group_size)
    codes = np.zeros((out_f, in_f), dtype=np.uint8)
    scales = np.zeros((out_f, n_groups), dtype=np.float32)
    for i1 in range(0, in_f, block_size):
        i2 = min(i1 + block_size, in_f)
        W1 = W[:, i1:i2].copy()
        E = np.zeros((out_f, i2 - i1))
        for i in range(i2 - i1):
            j = i1 + i
            g = j // group_size
            if j % group_size == 0:
                # current weights of the group: block-lazy updates not yet applied to W
                # are the in-block errors E[:, :i]
                e = min(j + group_size, in_f)
                cur = W[:, j:e] - E[:, :i] @ U[i1:j, j:e]
                scales[:, g] = _group_scales(cur[:, None, :], grid, signed_scale)[:, 0]
            s = scales[:, g].astype(np.float64)
            w = W1[:, i]
            c = _encode(w, s, grid)
            codes[:, j] = c
            err = (w - s * grid[c]) / U[j, j]
            W1[:, i:] -= np.outer(err, U[j, j:i2])
            E[:, i] = err
        W[:, i2:] -= E @ U[i1:i2, i2:]

    w_q = dequantize(codes, scales, bits, group_size)
    D = w_q.astype(np.float64) - W0
    err_out = 0.5 * float(np.sum((D @ H0) * D))
    return {"codes": codes, "scales": scales, "w_q": w_q, "err": err_out}


# ------------------------------------------------------------ App A inits ---

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


