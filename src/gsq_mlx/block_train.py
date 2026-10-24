"""Block-wise GSQ training skeleton (paper Sec 4.1, Apps C/G).

Staged schedule for dense blocks (paper "Within-block staging"):
  stage "qk"  (Q/K): independent linear reconstruction losses
  stage "vo"  (V/O): joint self-attention-output reconstruction loss
  stage "mlp" (gate/up/down): full block reconstruction loss, last

Each stage minimizes ||f(X; w_bar) - f(X; w)||^2_F over soft weights
w_bar = s * (p @ grid) from Gumbel-Softmax (tau 2 -> 0.05, kappa 100 -> 500,
linear) with per-(row, group) scales s, optimized with Lion (logits 1e-4,
scales 5e-5, wd 1.0, betas 0.9/0.95), with gradient accumulation over Gumbel
resamples (paper: resampling each micro-batch averages down sampling variance).

Phase 1 scope: shapes + losses + Lion steps + accumulation are REAL but tiny
(synthetic weights, synthetic-token calibration stub). Phase 2 wires: real
Qwen weights/activations, real tokenizer calibration tokens, frozen-prefix
block loop, scale-only fine-tune.

Two compute paths:
  *_np  numpy path used by scripts/test_gsq_phase1.py (fast, <30s)
  *_mx  mlx path — pure functions of (params, noise) so a loss closure is
        directly usable under mx.value_and_grad; Lion-mx steps in lion.py
"""
from __future__ import annotations
import numpy as np

try:
    import mlx.core as mx  # type: ignore
    _HAS_MLX = True
except Exception:
    mx = None  # type: ignore
    _HAS_MLX = False

from .gptq import init_logits_2bit_from_codes, quantize_rtn, synthetic_weights
from .lion import Lion, LR_LOGITS, LR_SCALES
from .quant import get_grid, gs_schedules, gs_softmax, sample_gumbel

# Paper within-block staging order.
STAGES = ("qk", "vo", "mlp")
STAGE_LAYERS = {
    "qk": ("q_proj", "k_proj"),
    "vo": ("v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


# ------------------------------------------------------- calibration stub ---

def synthetic_calibration(num_seqs: int, seq_len: int, hidden: int,
                          seed: int = 0) -> np.ndarray:
    """STUB (Phase 2: real tokenizer + FineWeb-Edu/OpenThoughts tokens).
    Returns synthetic 'token' activations X [M, hidden], M = num_seqs*seq_len.
    Callers keep M tiny (e.g. 32x8) so Phase-1 tests run in seconds."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((int(num_seqs) * int(seq_len), int(hidden)),
                              dtype=np.float32)


# ------------------------------------------------------------- numpy path ---

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


# ---------------------------------------------------------------- mlx path ---

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


def train_step_mx(params: dict, grads: dict, opt: Lion) -> dict:
    """One Lion-mx step (thin wrapper so staged loops read cleanly)."""
    return opt.step(params, grads)


def staged_block_plan(layer_names: tuple[str, ...] = STAGE_LAYERS["mlp"]):
    """Yield (stage, layers) in paper staging order for the requested layer
    subset. Phase 2 iterates this over real blocks with frozen prefixes."""
    for stage in STAGES:
        layers = tuple(n for n in STAGE_LAYERS[stage] if n in layer_names)
        if layers:
            yield stage, layers
