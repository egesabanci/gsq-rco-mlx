"""Block-wise staged GSQ on the real Qwen2 stream (paper Sec 3.3, 4.1, Apps A, C, H).

A block is quantized one at a time. Its seven linears go through the paper's three stages
(STAGE_LAYERS), in this order:
  qk   q_proj, k_proj   linear reconstruction ||X (W_bar - W)^T||^2, one loss per linear
  vo   v_proj, o_proj   self-attention output ||A_bar(x) - A(x)||^2 (q, k hard)
  mlp  gate, up, down   block output ||Y_bar(x) - Y(x)||^2 (attention hard)
Inside a stage, each linear first gets a GPTQ warm start (gptq.quantize_gptq). Its Hessian comes
from its real input: the calibration stream with every earlier stage hardened, and the earlier
linears of this stage at their GPTQ values. Then Gumbel-Softmax logits and group scales are trained
with Lion, and the stage is hardened to argmax codes and learned scales rounded to fp16, which is
the stored form.

Every loss and Hessian skips sequence position 0. In layers 2, 3 and 21 position 0 carries 96-99.6%
of the down_proj input energy (FINDINGS #10). Evaluation keeps it.

8-bit linears are GPTQ-only (256 logits per weight are out of budget): their codes and scales are
frozen in the stage optimiser, while App H still fine-tunes 8-bit scales. Ternary (bits=1) uses mask
and sign logits (Alg.2, alpha=3). 2-bit uses the full grid (Eq.5, alpha=6). 3..7-bit use the 5-logit
local shift (Eq.6, alpha=6, mu=0). Table 7 gives std=0.01, lr 1e-4 (logits) and 5e-5 (scales), wd
1.0, betas (0.9, 0.95), tau 2 -> 0.05 and kappa 100 -> 500 over each stage (quant.gs_schedules).
Group scales of 4..7-bit linears are frozen as well (lion.scale_lr, WP5c: above 3 bits a scale step
raised the stage objectives on real layer 3). App H still fine-tunes every scale.

Implementation notes (memory and speed, measured on layer 5):
  * tau and kappa are mx scalars, so the chunk functions take them as arrays. The gradient is
    eager: mx.compile of the whole step raised the MLX peak from about 1.5 GB to 2.0 GB.
  * The soft weights are built in row chunks of CHUNK_ROWS under mx.checkpoint: the Gumbel noise and
    softmax temporaries of a 13M-row MLP stage are otherwise all live in the backward pass (2.4 GB).
  * Gumbel noise is -log(-log u) with u from mx.random.uniform (same distribution, ~2x cheaper).
  * Lion is applied one tensor at a time (lion.lion_update, the one Lion rule of the package).
  * FP targets of a stage are computed once for all calibration sequences.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from gsq_rco.common.evaluate import kl_value_and_grad_hidden

from .e2e import LIN_ORDER, Qwen2Stream, rmsnorm, silu
from .gptq import (dequantize, init_logits_2bit_from_codes, init_shift_logits,
                   init_ternary_logits, quantize_gptq)
from .lion import LR_LOGITS, LR_SCALES, lion_update, scale_lr
from .quant import get_grid, gs_schedules, local_shift_indices, local_shift_values

ALPHA_TERNARY = 3.0   # Table 7, 1.58-bit
ALPHA_SCALAR = 6.0    # Table 7, 2/3-bit; also the local shift for 3..7 bits
INIT_STD = 0.01       # Table 7, std
CHUNK_ROWS = 1 << 20  # logit rows per checkpointed chunk

# Paper within-block staging order (Sec 4.1, App C).
STAGES = ("qk", "vo", "mlp")
STAGE_LAYERS = {
    "qk": ("q_proj", "k_proj"),
    "vo": ("v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
}


@dataclass(frozen=True)
class BlockOpts:
    """Optimisation settings for one block. Defaults follow App G, with a calibration set of
    128 x 512 tokens. steps is the Lion step count per stage (see the WP5 report for the budget)."""
    group_size: int = 64
    damp: float = 0.01          # GPTQ damping, relative to mean(diag H)
    block_size: int = 128       # GPTQ lazy-update block
    steps: int = 200            # Lion steps per stage; paper ~1280 (budget: see the WP5 report)
    micro_batch: int = 2        # calibration sequences per step
    resamples: int = 2          # independent Gumbel draws whose gradients are averaged per step
    lr_logits: float = LR_LOGITS
    lr_scales: float = LR_SCALES
    seed: int = 0


# ------------------------------------------------------- gumbel-softmax ---

def _gs_softmax(logits, g, tau, kappa):
    """Alg.1 with array-valued tau and kappa: p = softmax((kappa * l + g) / tau)."""
    return mx.softmax((kappa * logits + g) / tau, axis=-1)


def _binary_gs(logit, g0, g1, tau, kappa):
    """Binary Alg.1 (Alg.2 lines 4-5) for the pair {-l, +l}: P(active) = sigmoid((2 kappa l + g1 - g0) / tau)."""
    return mx.sigmoid((2.0 * kappa * logit + (g1 - g0)) / tau)


def _gumbel(key, shape):
    """Standard Gumbel(0, 1) noise, -log(-log u) with u ~ U(0, 1). The key carries no gradient;
    stop_gradient keeps mx.checkpoint from differentiating the RNG primitive."""
    u = mx.random.uniform(shape=shape, key=mx.stop_gradient(key))
    return -mx.log(-mx.log(mx.clip(u, 1e-7, 1.0 - 1e-7)))


def _grid_rows(L, key, tau, kappa, grid):
    """c = p @ grid for one chunk of full-grid logits (Alg.1, Eq.5)."""
    return _gs_softmax(L, _gumbel(key, L.shape), tau, kappa) @ grid


def _shift_rows(L, C, key, tau, kappa):
    """c = sum_k p_k C_k for one chunk of local-shift logits (Eq.6); C holds the candidate values."""
    return mx.sum(_gs_softmax(L, _gumbel(key, L.shape), tau, kappa) * C, axis=-1)


def _ternary_rows(lm, lb, key, tau, kappa):
    """c = m * b for one chunk of mask and sign logits (Alg.2 lines 4-7)."""
    k = mx.random.split(mx.stop_gradient(key), 4)
    m = _binary_gs(lm, _gumbel(k[0], lm.shape), _gumbel(k[1], lm.shape), tau, kappa)
    b = 2.0 * _binary_gs(lb, _gumbel(k[2], lb.shape), _gumbel(k[3], lb.shape), tau, kappa) - 1.0
    return m * b


_grid_rows_ck = mx.checkpoint(_grid_rows)
_shift_rows_ck = mx.checkpoint(_shift_rows)
_ternary_rows_ck = mx.checkpoint(_ternary_rows)


# --------------------------------------------------------------- helpers ---

def _take(x, idx):
    return mx.take(x, mx.array(np.asarray(idx, dtype=np.int32)), axis=0)


def _rows(x):
    """[c, T, d] -> [c*(T-1), d] float32 over positions 1..T-1 (position 0 excluded)."""
    return x[:, 1:].reshape(-1, x.shape[-1]).astype(mx.float32)


def _per_seq(n, fn, eval_every: int = 8):
    """Stack fn(s) over sequences s < n, evaluating every few sequences to bound the graph."""
    outs = []
    for s in range(n):
        outs.append(fn(s))
        if (s + 1) % eval_every == 0 or s == n - 1:
            mx.eval(*outs[-eval_every:])
    return mx.stack(outs)


def hessian_add(H, n, X):
    """One streaming step of gptq.hessian_from_inputs (H = 2 X^T X / M, float64). The Gram matrix
    is formed on the device in float32, because a float64 CPU product over 4864-dim inputs costs
    about 30 s per block. X: mx [m, d]. Returns (H, n + m)."""
    total = n + X.shape[0]
    G = np.array(X.T @ X, dtype=np.float64)
    if H is None:
        H = np.zeros_like(G)
    else:
        H *= n / total
    G *= 2.0 / total
    H += G
    return H, total


def _hessian(n_seqs, rows_of, chunk=8):
    """Hessian over calibration rows; rows_of(a, b) returns the rows of sequences a..b-1."""
    H, n = None, 0
    for a in range(0, n_seqs, chunk):
        H, n = hessian_add(H, n, rows_of(a, min(a + chunk, n_seqs)))
    return H


def _attn_rows(stream, W, hb, eps):
    """o_proj inputs (attention before o_proj) for a chunk of sequences, positions 1..T-1."""
    return mx.concatenate([stream.attn_core(rmsnorm(hb[s], W["in_norm"], eps), W)[1:]
                           for s in range(hb.shape[0])], axis=0).astype(mx.float32)


def _mlp_hidden(stream, W, hb, eps):
    """Post-attention residual h1 and its post_norm, [c, T, H] each, under W."""
    h1s, mns = [], []
    for s in range(hb.shape[0]):
        o = stream.attn_core(rmsnorm(hb[s], W["in_norm"], eps), W)
        h1 = hb[s] + o @ W["o_proj"].T
        h1s.append(h1)
        mns.append(rmsnorm(h1, W["post_norm"], eps))
    return mx.stack(h1s), mx.stack(mns)


def _down_rows(stream, W, hb, eps):
    """down_proj inputs (silu(gate) * up) for a chunk of sequences, positions 1..T-1, under W."""
    mn = _mlp_hidden(stream, W, hb, eps)[1]
    return _rows(silu(mn @ W["gate_proj"].T) * (mn @ W["up_proj"].T))


def _wq_np(codes, scales16, bits, group_size):
    """Stored form (uint8 codes, fp16 scales) -> dequantized float32 weight, numpy."""
    return dequantize(codes, scales16.astype(np.float32), bits, group_size)


def _hard_bf16(codes, scales16, bits, group_size):
    """Stored form -> the bf16 weight that the stream and the mlx-lm export both use."""
    return mx.array(_wq_np(codes, scales16, bits, group_size)).astype(mx.bfloat16)


def _lin_mse(Wq, Wf, H):
    """Mean squared calibration output error per output element of one linear:
    0.5 tr(D H D^T) / out with D = Wq - W (the gptq.quantize_gptq err, divided by out)."""
    D = Wq.astype(np.float64) - Wf.astype(np.float64)
    return 0.5 * float(np.sum((D @ H) * D)) / D.shape[0]


def _attn_mse(stream, W, h, A_fp, eps):
    """Mean squared self-attention output error (after o_proj) against precomputed targets
    A_fp [N, T, H], positions 1..T-1, over all sequences."""
    tot, cnt = 0.0, 0
    for s in range(h.shape[0]):
        A = stream.attn_core(rmsnorm(h[s], W["in_norm"], eps), W) @ W["o_proj"].T
        D = (A - A_fp[s])[1:].astype(mx.float32)
        tot += float(mx.sum(D * D))
        cnt += D.size
    return tot / cnt


def _block_mse(stream, W, h, Y_fp):
    """Mean squared block output error against precomputed targets Y_fp, positions 1..T-1."""
    tot, cnt = 0.0, 0
    for s in range(h.shape[0]):
        D = (stream.block(h[s], W) - Y_fp[s])[1:].astype(mx.float32)
        tot += float(mx.sum(D * D))
        cnt += D.size
    return tot / cnt


# ----------------------------------------------------------- linear state ---

class _Linear:
    """One linear under the stage optimiser: GPTQ codes and scales, plus trainable logits.

    Logits per weight are 2 (bits=2, full grid, Eq.5), 5 (bits 3..7, local shift, Eq.6) or two
    binary logits (bits=1, mask and sign, Alg.2). Group scales [out, G] enter through the group
    index of every input column (``gid``). bits=8 is GPTQ only: no logits, and in the stage optimiser
    its codes and scales stay fixed, so its weight is the constant ``w8``. The arrays of self.P are
    plain keys ("logits", "lm", "lb", "scales"); during training they are prefixed with the
    linear's name.
    """

    def __init__(self, name, bits, codes, scales16, group_size, seed):
        self.name, self.bits = name, bits
        self.shape = codes.shape
        self.codes0 = codes
        self.scales16 = scales16
        self.grid = get_grid(bits)
        self.grid_mx = mx.array(self.grid)
        self.gid = mx.array((np.arange(self.shape[1]) // group_size).astype(np.int32))
        self.P = {"scales": mx.array(scales16.astype(np.float32))}
        d = self.shape[0] * self.shape[1]
        if bits == 1:
            lm, lb = init_ternary_logits(self.grid[codes].reshape(-1), alpha=ALPHA_TERNARY,
                                         std=INIT_STD, seed=seed)
            self.P["lm"], self.P["lb"] = mx.array(lm), mx.array(lb)
        elif bits == 2:
            self.P["logits"] = mx.array(init_logits_2bit_from_codes(
                codes, len(self.grid), alpha=ALPHA_SCALAR, std=INIT_STD, seed=seed))
        elif bits == 8:
            # Dequantized GPTQ weight, scales[:, gid] * grid[codes] (the formula of gptq.dequantize).
            self.w8 = mx.take(self.P["scales"], self.gid, axis=1) * mx.take(
                self.grid_mx, mx.array(codes.astype(np.int32)), axis=0)
        else:
            self.P["logits"] = mx.array(init_shift_logits(
                d, alpha=ALPHA_SCALAR, std=INIT_STD, seed=seed))
            self.C = mx.array(local_shift_values(
                codes.reshape(-1).astype(np.int64), self.grid).astype(np.float32))

    def key(self, p: str) -> str:
        return f"{self.name}.{p}"

    def soft(self, P, key, tau, kappa):
        """Soft weight w_bar = s * c in fp32 [out, in]. c is one Gumbel-Softmax sample of the grid
        value, built chunk by chunk (Alg.1: c = p @ grid; Eq.6 for shifts; Alg.2 for ternary,
        c = m * b). s is the group scale. ``key`` is split into one key per chunk; tau and kappa
        are mx scalars. bits=8 returns the fixed GPTQ weight and ignores P, key, tau and kappa."""
        if self.bits == 8:
            return self.w8
        d = self.shape[0] * self.shape[1]
        bounds = [(a, min(a + CHUNK_ROWS, d)) for a in range(0, d, CHUNK_ROWS)]
        keys = mx.random.split(key, len(bounds))
        parts = []
        for j, (a, b) in enumerate(bounds):
            if self.bits == 1:
                parts.append(_ternary_rows_ck(P[self.key("lm")][a:b], P[self.key("lb")][a:b],
                                              keys[j], tau, kappa))
            elif self.bits == 2:
                parts.append(_grid_rows_ck(P[self.key("logits")][a:b], keys[j], tau, kappa,
                                           self.grid_mx))
            else:
                parts.append(_shift_rows_ck(P[self.key("logits")][a:b], self.C[a:b], keys[j],
                                            tau, kappa))
        c = parts[0] if len(parts) == 1 else mx.concatenate(parts)
        # NOTE on the expansion of the group scales to columns (mx.take): its backward is a scatter-add over
        # groups. On the MLX GPU the accumulation order of that scatter is not fixed, so the scale gradient,
        # and in rare cases a code flip, can differ between two runs at the last bits (measured: repeated
        # evaluations of this vjp disagree in about 80% of the elements; two GPU runs of the same settings
        # disagree in about 21% of the gate codes). The expansion is kept, so the numerics are unchanged.
        return mx.take(P[self.key("scales")], self.gid, axis=1) * c.reshape(self.shape)

    def hard_codes(self, P) -> np.ndarray:
        """Argmax codes of the learned logits (Alg.2 line 11; Eq.6 shifts, clipped to the grid).
        bits=8: the GPTQ codes, frozen."""
        if self.bits == 8:
            return self.codes0
        if self.bits == 1:
            m = np.asarray(P["lm"]) >= 0
            b = np.asarray(P["lb"]) >= 0
            codes = m.astype(np.int64) * (2 * b.astype(np.int64) - 1) + 1
        elif self.bits == 2:
            codes = np.asarray(P["logits"]).argmax(axis=-1)
        else:
            codes = local_shift_indices(self.codes0.reshape(-1).astype(np.int64),
                                        np.asarray(P["logits"]), len(self.grid))
        return np.asarray(codes, dtype=np.uint8).reshape(self.shape)

    def stored_scales(self, P) -> np.ndarray:
        return np.asarray(P["scales"], dtype=np.float32).astype(np.float16)


# ------------------------------------------------------------ stage losses ---

def _loss_linear(lins, targets, P, X, key, tau, kappa):
    """qk: sum over linears of mean ||X (W_bar - W)^T||^2 (the linear output error)."""
    keys = mx.random.split(key, len(lins))
    total = 0.0
    for i, lin in enumerate(lins):
        R = X @ (lin.soft(P, keys[i], tau, kappa) - targets[lin.name]).T
        total = total + mx.mean(R * R)
    return total


def _loss_attn(stream, W, lins, P, data, key, tau, kappa):
    """vo: mean ||A_bar - A||^2 over positions 1..T-1, A = self-attention output after o_proj."""
    HN, A_fp = data
    keys = mx.random.split(key, 2)
    Wd = dict(W, v_proj=lins[0].soft(P, keys[0], tau, kappa))
    Wo = lins[1].soft(P, keys[1], tau, kappa)
    total = 0.0
    for s in range(HN.shape[0]):
        R = (stream.attn_core(HN[s], Wd) @ Wo.T - A_fp[s])[1:]
        total = total + mx.mean(R * R)
    return total / HN.shape[0]


def _loss_block(stream, W, lins, P, data, key, tau, kappa):
    """mlp: mean ||Y_bar - Y||^2 over positions 1..T-1, Y = block output, attention hard."""
    Hin, Y_fp = data
    keys = mx.random.split(key, len(lins))
    Wd = dict(W, **{lin.name: lin.soft(P, keys[i], tau, kappa) for i, lin in enumerate(lins)})
    total = 0.0
    for s in range(Hin.shape[0]):
        R = (stream.block(Hin[s], Wd) - Y_fp[s])[1:]
        total = total + mx.mean(R * R)
    return total / Hin.shape[0]


def _train(lins, data_of, loss_of, n_seqs, opts, seed):
    """Stage optimiser (paper Sec 3.3). Each step samples micro_batch sequences, averages the
    gradients over ``resamples`` independent Gumbel draws, then takes one Lion step on the logits
    (lr_logits) and one on the group scales (lion.scale_lr(lr_scales, bits): zero above 3 bits).
    tau and kappa follow App G across the stage.
    loss_of(P, data, key, tau, kappa) gives the scalar soft loss. Writes the trained arrays back
    into each _Linear.P and returns the soft loss of each step. bits=8 linears are not trained
    (codes and scales frozen); a stage with no other linear runs no steps."""
    trainable = {lin.key(p): v for lin in lins if lin.bits != 8 for p, v in lin.P.items()}
    if not trainable or opts.steps == 0:
        return [], None
    lr = {lin.key(p): mx.array(scale_lr(opts.lr_scales, lin.bits) if p == "scales" else opts.lr_logits)
          for lin in lins if lin.bits != 8 for p in lin.P}
    mom = {k: mx.zeros_like(v) for k, v in trainable.items()}
    R = opts.resamples
    update = mx.compile(lion_update)
    rng = np.random.default_rng(seed)
    m = min(opts.micro_batch, n_seqs)
    history, step_s = [], []
    for t in range(opts.steps):
        t0 = time.perf_counter()
        tau_t, kappa_t = gs_schedules(t, opts.steps)
        tau, kappa = mx.array(tau_t), mx.array(kappa_t)
        data = data_of(np.sort(rng.choice(n_seqs, size=m, replace=False)))
        # Eager autodiff, not mx.compile: compiling the step raised the MLX peak from 1.5 to 2.0 GB.
        acc, loss_sum = None, 0.0
        for r in range(R):
            key = mx.random.key(seed * 100_003 + t * R + r)
            loss, g = mx.value_and_grad(lambda Q: loss_of(Q, data, key, tau, kappa))(trainable)
            acc = g if acc is None else {k: acc[k] + g[k] for k in g}
            loss_sum += float(loss)
            mx.eval(acc)
        for k in trainable:  # one tensor at a time: old and new Lion state never coexist in full
            new_p, new_m = update({k: trainable[k]}, {k: mom[k]}, {k: acc[k] / R}, {k: lr[k]})
            trainable[k], mom[k] = new_p[k], new_m[k]
            mx.eval(trainable[k], mom[k])
        del acc
        history.append(loss_sum / R)
        step_s.append(time.perf_counter() - t0)
    for lin in lins:
        if lin.bits != 8:
            for p in lin.P:
                lin.P[p] = trainable[lin.key(p)]
    return history, round(1000 * float(np.median(step_s[3:] or step_s)), 1)


# ---------------------------------------------------------- block quantizer ---

def quantize_block(stream: Qwen2Stream, h, W: dict, bits: dict, opts: BlockOpts,
                   layer: int, log=print):
    """Quantize the seven linears of one block on the calibration stream.

    h: bf16 [N, T, H], the inputs of this block for each calibration sequence (the quantized
    prefix). W: FP block weights (linears, q/k/v biases, in_norm, post_norm). bits: {linear: width}.
    Returns (shard, stats, Wq). shard[name] = {"codes": uint8 [out, in], "scales": float16
    [out, G], "bits": int}. Wq holds the bf16 hard weights of the whole block (the next stream's
    weights). stats has per-stage timing, objectives and per-linear diagnostics.
    """
    N, T, _ = h.shape
    eps, gs = stream.eps, opts.group_size
    Wf = {name: np.asarray(W[name].astype(mx.float32)) for name in LIN_ORDER}
    Wfm = {name: W[name].astype(mx.float32) for name in LIN_ORDER}
    Wcur = dict(W)  # hardened so far; GPTQ values of the current stage while it is set up
    seed = opts.seed * 1_000_003 + layer * 1_009
    shard, stats = {}, {"stages": {}, "linears": {}}

    def gptq(name, H, pos):
        g = quantize_gptq(Wf[name], H, bits[name], gs, opts.damp, opts.block_size)
        codes0, s16 = g["codes"], g["scales"].astype(np.float16)
        Wcur[name] = _hard_bf16(codes0, s16, bits[name], gs)
        return _Linear(name, bits[name], codes0, s16, gs, seed=seed + 10 * pos), H

    def harden(lin, H):
        codes = lin.hard_codes(lin.P)
        s16 = lin.stored_scales(lin.P)
        Wcur[lin.name] = _hard_bf16(codes, s16, lin.bits, gs)
        shard[lin.name] = {"codes": codes, "scales": s16, "bits": lin.bits}
        ls = stats["linears"][lin.name]
        ls["flip_frac"] = float(np.mean(codes != lin.codes0))
        ls["mse_hard_final"] = _lin_mse(_wq_np(codes, s16, lin.bits, gs), Wf[lin.name], H)

    def init_stats(lin, H):
        stats["linears"].setdefault(lin.name, {"bits": lin.bits})["mse_gptq_init"] = _lin_mse(
            _wq_np(lin.codes0, lin.scales16, lin.bits, gs), Wf[lin.name], H)

    def stage_done(stage, t0, obj0, obj1, hist, median_step_ms):
        stats["stages"][stage] = {
            "seconds": round(time.perf_counter() - t0, 2),
            "objective_init": obj0, "objective_final": obj1,
            "soft_loss_first50": float(np.mean(hist[:50])) if hist else None,
            "soft_loss_last50": float(np.mean(hist[-50:])) if hist else None,
            "steps": len(hist), "median_step_ms": median_step_ms,
        }
        timing = f"median {median_step_ms} ms/step" if hist else "no Lion steps"
        log(f"    stage {stage}: objective {obj0:.4e} -> {obj1:.4e} "
            f"({stats['stages'][stage]['seconds']:.1f}s, {len(hist)} steps, {timing})")

    # ---- qk: linear reconstruction, one loss per linear ----------------
    t0 = time.perf_counter()
    H_hn = _hessian(N, lambda a, b: _rows(rmsnorm(h[a:b], W["in_norm"], eps)))
    qk = [gptq(n, H_hn, i) for i, n in enumerate(STAGE_LAYERS["qk"])]
    qk_lins = [lin for lin, _ in qk]
    for lin, H in qk:
        stats["linears"][lin.name] = {"bits": lin.bits}
        init_stats(lin, H)
    obj0 = float(np.mean([stats["linears"][lin.name]["mse_gptq_init"] for lin in qk_lins]))
    hist, ms = _train(qk_lins,
                  lambda idx: _rows(rmsnorm(_take(h, idx), W["in_norm"], eps)),
                  lambda P, X, key, tau, kappa: _loss_linear(qk_lins, Wfm, P, X, key, tau, kappa),
                  N, opts, seed + 1)
    for lin, H in qk:
        harden(lin, H)
    stage_done("qk", t0, obj0,
               float(np.mean([stats["linears"][lin.name]["mse_hard_final"] for lin in qk_lins])),
               hist, ms)

    # ---- vo: self-attention output, q and k hard -----------------------
    t0 = time.perf_counter()
    v_lin = gptq("v_proj", H_hn, 2)[0]
    H_o = _hessian(N, lambda a, b: _attn_rows(stream, Wcur, h[a:b], eps))
    o_lin = gptq("o_proj", H_o, 3)[0]
    vo = [v_lin, o_lin]
    Hs = {"v_proj": H_hn, "o_proj": H_o}
    for lin in vo:
        stats["linears"].setdefault(lin.name, {"bits": lin.bits})
        init_stats(lin, Hs[lin.name])
    A_all = _per_seq(N, lambda s: stream.attn_core(rmsnorm(h[s], W["in_norm"], eps), W)
                     @ W["o_proj"].T).astype(mx.bfloat16)
    obj0 = _attn_mse(stream, Wcur, h, A_all, eps)
    Wd_vo = dict(Wcur)
    hist, ms = _train(vo,
                  lambda idx: (rmsnorm(_take(h, idx), W["in_norm"], eps), _take(A_all, idx)),
                  lambda P, data, key, tau, kappa: _loss_attn(stream, Wd_vo, vo, P, data,
                                                               key, tau, kappa),
                  N, opts, seed + 2)
    for lin in vo:
        harden(lin, Hs[lin.name])
    stage_done("vo", t0, obj0, _attn_mse(stream, Wcur, h, A_all, eps), hist, ms)
    del A_all

    # ---- mlp: block output, attention hard -----------------------------
    t0 = time.perf_counter()
    H_gu = _hessian(N, lambda a, b: _rows(_mlp_hidden(stream, Wcur, h[a:b], eps)[1]))
    gate, up = gptq("gate_proj", H_gu, 4)[0], gptq("up_proj", H_gu, 5)[0]
    H_d = _hessian(N, lambda a, b: _down_rows(stream, Wcur, h[a:b], eps))
    down = gptq("down_proj", H_d, 6)[0]
    mlp = [gate, up, down]
    Hs.update({"gate_proj": H_gu, "up_proj": H_gu, "down_proj": H_d})
    for lin in mlp:
        stats["linears"].setdefault(lin.name, {"bits": lin.bits})
        init_stats(lin, Hs[lin.name])
    Y_all = _per_seq(N, lambda s: stream.block(h[s], W)).astype(mx.bfloat16)
    obj0 = _block_mse(stream, Wcur, h, Y_all)
    Wd_mlp = dict(Wcur)
    hist, ms = _train(mlp,
                  lambda idx: (_take(h, idx), _take(Y_all, idx)),
                  lambda P, data, key, tau, kappa: _loss_block(stream, Wd_mlp, mlp, P, data,
                                                                key, tau, kappa),
                  N, opts, seed + 3)
    for lin in mlp:
        harden(lin, Hs[lin.name])
    stage_done("mlp", t0, obj0, _block_mse(stream, Wcur, h, Y_all), hist, ms)
    del Y_all

    stats["peak_mlx_bytes"] = int(mx.get_peak_memory())
    return shard, stats, Wcur


def hard_block(lins: dict, base: dict, group_size: int) -> dict:
    """bf16 weights of a quantized block from its stored form. lins[name] = {"codes" uint8,
    "scales" float16, "bits"}; base holds the FP biases and norms of the block."""
    W = dict(base)
    for name in LIN_ORDER:
        q = lins[name]
        W[name] = _hard_bf16(q["codes"], q["scales"], q["bits"], group_size)
    return W


def advance(stream: Qwen2Stream, h, W: dict):
    """Run one block over every sequence of the stream, each sequence once: [N, T, H] -> [N, T, H]."""
    return _per_seq(h.shape[0], lambda s: stream.block(h[s], W))


# ----------------------------------------------------- scale-only fine-tune ---

class _TiedEmbed:
    def __init__(self, E):
        self.weight = E

    def as_linear(self, h):
        return h @ self.weight.T


def tied_head(E):
    """Tied LM head over the embedding E [V, H], in the form gsq_rco.common.evaluate expects."""
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=_TiedEmbed(E)))


def scale_finetune(stream: Qwen2Stream, qblocks: dict, base: dict, E, final_norm, tokens,
                   h_ref, steps: int, lr: float, group_size: int = 64, log=print):
    """App H: end-to-end scale-only fine-tune. Codes stay fixed. Every group scale of every
    quantized layer is trained by Lion against KL(p_fp || p_q) on one calibration sequence per step.
    The KL is chunked and memory-bounded (evaluate.kl_value_and_grad_hidden) and is chained through
    the network with mx.vjp. Every block is wrapped in mx.checkpoint and, if quantized, dequantized
    inside it, so only one block's bf16 weights and activations are live at a time.

    qblocks[layer][name] = {"codes" uint8, "scales" float16, "bits"} for the quantized layers.
    base[layer] holds the FP biases and norms of a quantized layer, or the whole FP block of an
    unquantized one. E and final_norm are the FP embedding and final norm, tokens [N, T] int, and
    h_ref [N, T, H] the final-normed FP hidden states. Returns (scales, kl): scales[layer][name]
    are the fine-tuned fp16 scales, kl the KL of each step before its update.
    """
    n_layers = len(base)
    keys, grids, gids, codes, flat = [], {}, {}, {}, []
    for li in sorted(qblocks):
        for name in LIN_ORDER:
            q = qblocks[li][name]
            _, in_f = q["codes"].shape
            keys.append((li, name))
            codes[(li, name)] = mx.array(q["codes"])
            grids[(li, name)] = mx.array(get_grid(q["bits"]))
            gids[(li, name)] = mx.array((np.arange(in_f) // group_size).astype(np.int32))
            flat.append(mx.array(q["scales"].astype(np.float32)))

    def make_block(li):
        def fn(h, *scales):
            W = dict(base[li])
            for name, sc in zip(LIN_ORDER, scales):
                k = (li, name)
                # Same expansion and same GPU caveat as _Linear.soft (mx.take backward is a scatter-add).
                W[name] = (mx.take(sc, gids[k], axis=1)
                           * mx.take(grids[k], codes[k].astype(mx.int32), axis=0)
                           ).astype(mx.bfloat16)
            return stream.block(h, W)
        return mx.checkpoint(fn)

    def make_fp_block(li):
        return mx.checkpoint(lambda h: stream.block(h, base[li]))

    blocks = {li: (make_block(li) if li in qblocks else make_fp_block(li))
              for li in range(n_layers)}

    def forward(*scales_flat, tok):
        h = mx.take(E, mx.array(tok), axis=0)
        k = 0
        for li in range(n_layers):
            if li in qblocks:
                h = blocks[li](h, *scales_flat[7 * k:7 * k + 7])
                k += 1
            else:
                h = blocks[li](h)
        return rmsnorm(h, final_norm)[None]

    head = tied_head(E)
    slots = [str(i) for i in range(len(flat))]
    params = dict(zip(slots, flat))
    mom = {s: mx.zeros_like(v) for s, v in params.items()}
    lr_map = {s: mx.array(lr) for s in slots}
    update = mx.compile(lion_update)
    kls = []
    for t in range(steps):
        n = t % tokens.shape[0]
        tok = tokens[n]
        h_q = forward(*[params[s] for s in slots], tok=tok)
        kl, g = kl_value_and_grad_hidden(head, h_q, h_ref[n:n + 1])
        mx.eval(kl, g)
        _, grads = mx.vjp(lambda *a: forward(*a, tok=tok),
                          [params[s] for s in slots], [g])
        params, mom = update(params, mom, dict(zip(slots, grads)), lr_map)
        mx.eval(params, mom)
        kls.append(float(kl))
        if (t + 1) % max(1, steps // 4) == 0:
            log(f"    scale fine-tune step {t + 1}/{steps}: KL "
                f"{np.mean(kls[-max(1, steps // 4):]):.5f}")
    out = {}
    for s, (li, name) in zip(slots, keys):
        out.setdefault(li, {})[name] = np.asarray(params[s], dtype=np.float32).astype(np.float16)
    return out, kls
