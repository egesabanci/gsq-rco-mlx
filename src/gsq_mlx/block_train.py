"""Block-wise GSQ training skeleton (paper Sec 4.1, Apps C/G).

Staged schedule for dense blocks (paper "Within-block staging"):
  stage "qk"  (Q/K): independent linear reconstruction losses
  stage "vo"  (V/O): joint self-attention-output reconstruction loss
  stage "mlp" (gate/up/down): full block reconstruction loss, last

Each stage minimizes ||f(X; w_bar) - f(X; w)||^2_F over soft weights
w_bar from Gumbel-Softmax (tau 2 -> 0.05, kappa 100 -> 500, linear),
optimized with Lion (lr 1e-4 logits, 2e-5 scales, wd 1.0, betas
0.9/0.95), with gradient accumulation over Gumbel resamples (paper:
resampling each micro-batch averages down sampling variance).

Phase 1 scope: shapes + losses + Lion steps + accumulation are REAL but
tiny (synthetic weights, synthetic-token calibration stub). Phase 2 wires:
real Qwen weights/activations, real tokenizer calibration tokens,
frozen-prefix block loop, scale-only fine-tune.

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

from .lion import Lion, LR_LOGITS, LR_SCALES_QWEN
from .quant import (gs_schedules, gs_softmax, sample_gumbel,
                    quantize_soft, TAU_START, TAU_END,
                    KAPPA_START, KAPPA_END)

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


def soft_forward_np(logits: np.ndarray, grid: np.ndarray, scale: float,
                    tau: float, kappa: float,
                    g: np.ndarray | None = None,
                    rng: np.random.Generator | None = None):
    """Soft weights + probs for a flat d-vector of weights."""
    if g is None:
        g = sample_gumbel(logits.shape, rng=rng or np.random.default_rng(0))
    p = gs_softmax(np.asarray(logits), np.asarray(g), tau, kappa)
    wq = quantize_soft(p, grid, scale).reshape(-1)
    return wq, p


def linear_loss_and_grads_np(X: np.ndarray, W: np.ndarray,
                             logits: np.ndarray, grid: np.ndarray,
                             scale: float, tau: float, kappa: float,
                             n_resamples: int = 4, seed: int = 0):
    """MSE + grads w.r.t. (logits, scale), averaged over ``n_resamples``
    Gumbel draws (paper: accumulation damps sampling variance).
    Analytic chain rule through the softmax Jacobian; scale via autograd-free
    scalar derivative. All tiny-array numpy."""
    rng = np.random.default_rng(seed)
    W = np.asarray(W, dtype=np.float32)
    d, K = np.asarray(logits).shape
    assert d == W.size, "logits rows must match W elements"
    gridv = np.asarray(grid, dtype=np.float32)
    acc_l, acc_gs, acc_glog = 0.0, 0.0, None
    n_out = W.shape[0]
    for _ in range(n_resamples):
        g = sample_gumbel((d, K), rng=rng)
        p = gs_softmax(np.asarray(logits), g, tau, kappa)  # [d,K]
        c = p @ gridv                                       # [d]
        Wq = (scale * c).reshape(W.shape)                  # [out,in]
        D = Wq - W
        R = X @ D.T                                         # [M,out]
        M = X.shape[0]
        loss = float((R * R).mean())
        G = (2.0 / (M * n_out)) * (R.T @ X)                # dL/dWq
        gwq = G.reshape(-1)                                 # [d]
        gc = scale * gwq                                    # dL/dc
        # softmax Jacobian per row: dp/dl = kappa/tau * (diag(p)-p p^T)
        s = kappa / max(tau, 1e-6)
        glog = s * p * (gc[:, None] - (gc[:, None] * p).sum(axis=1, keepdims=True))
        gs = float((G * (c.reshape(W.shape))).sum())
        acc_l += loss
        acc_gs += gs
        acc_glog = glog if acc_glog is None else acc_glog + glog
    return (acc_l / n_resamples,
            acc_glog / n_resamples,
            acc_gs / n_resamples)


def train_linear_synthetic(n_bits: int = 2, in_f: int = 16, out_f: int = 8,
                           group_size: int = 64, steps: int = 30,
                           n_resamples: int = 4, seed: int = 0,
                           warm_start: bool = True, sched_T: int | None = None,
                           scale_mult: float = 1.0, verbose: bool = False) -> dict:
    """Phase-1 demo: Lion on (logits, scale) for one synthetic linear layer.
    Proves the loop decreases the staged linear loss. Returns history.
    ``warm_start=False`` starts from flat logits (cold) to show full
    optimization progress; ``sched_T`` stretches the tau/kappa horizon
    past ``steps`` so short demos keep gradients flowing."""
    from .quant import get_grid
    from .gptq import synthetic_weights, rtn_quantize_groupwise
    grid = get_grid(n_bits)
    W = synthetic_weights((out_f, in_f), seed=seed + 1)
    X = synthetic_calibration(4, 8, in_f, seed=seed + 2)
    rtn = rtn_quantize_groupwise(W, grid, group_size)
    scale = float(rtn["scales"].mean()) * float(scale_mult)
    from .gptq import init_logits_2bit_from_codes
    if warm_start:
        logits = init_logits_2bit_from_codes(rtn["codes"], len(grid),
                                             seed=seed + 3).astype(np.float32)
    else:
        crng = np.random.default_rng(seed + 3)
        logits = (0.01 * crng.standard_normal(
            (W.size, len(grid))).astype(np.float32))
    opt = Lion({"logits": logits, "scale": np.array(scale, dtype=np.float32)},
               lr=LR_LOGITS)
    opt_s = Lion({"scale": np.array(scale, dtype=np.float32)},
                 lr=LR_SCALES_QWEN)
    hist = []
    params = {"logits": logits, "scale": np.array(scale, dtype=np.float32)}
    for t in range(steps):
        tau, kappa = gs_schedules(t, sched_T or steps)
        loss, glog, gs = linear_loss_and_grads_np(
            X, W, params["logits"], grid, float(params["scale"]),
            tau, kappa, n_resamples=n_resamples, seed=seed + 1000 + t)
        hist.append(loss)
        params = opt.step({"logits": params["logits"]},
                          {"logits": glog}) | {"scale": params["scale"]}
        # scale uses its own (smaller) lr, decoupled like the paper
        sc = opt_s.step({"scale": params["scale"]},
                        {"scale": np.array(gs, dtype=np.float32)})
        params["scale"] = sc["scale"]
        if verbose:
            print(f"  step {t:02d} tau={tau:.3f} kappa={kappa:.0f} loss={loss:.6f}")
    return {"history": hist, "params": params, "grid": grid, "W": W, "X": X}


# ---------------------------------------------------------------- mlx path ---

def linear_soft_loss_mx(logits, scale, g, X, Wflat, grid, tau: float,
                        kappa: float):
    """Pure MLX staged linear loss: mean((X@(wq-wo))^2), wq from GS probs.
    ``g`` (Gumbel noise) is an explicit arg, so a closure over
    (logits, scale) with fixed (g, X, Wflat, grid) is usable under
    ``mx.value_and_grad(..., argnums=(0, 1))``. Weight matrices are flat
    [d]; the output dim is recovered from X's input dim."""
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
