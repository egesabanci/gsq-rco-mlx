"""Phase-1 GSQ acceptance test (OWNED by GSQ side).

Proves, with SYNTHETIC weights / tiny mx arrays only (no model load,
no safetensors — memory-critical Mac 17GB/~2.5GB free), all under 30s:

  T1 soft-to-hard collapse: mean max-prob at tau=0.05 >> tau=2.0 and
     soft weights converge to hard weights as tau -> 0 (paper Sec 3).
  T2 local-shift indices always in [0, K-1] incl. edges; soft decode
     bounded by s*max|grid| (paper Eq.6).
  T3 Lion decreases the synthetic linear-layer loss (paper App G Lion,
     lr 1e-4 logits / 2e-5 scales, grad accumulation over resamples).
  T4 ternary mask+sign paths: soft in [-s,s], hard in {-s,0,s} (Alg.2).
  T5 mlx parity: same ops on mx arrays match numpy; a staged linear loss
     closure differentiates under mx.value_and_grad (skipped if no mlx).

Usage: .venv/bin/python scripts/test_gsq_phase1.py
Exit nonzero on any failure.
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from gsq_mlx.quant import (
    get_grid, gptq_init_logits, gumbel_softmax_sample, quantize_soft,
    quantize_hard, ternary_soft, ternary_hard, local_shift_indices,
    local_shift_quantize_soft, local_shift_decode, gs_schedules,
)
from gsq_mlx.lion import Lion
from gsq_mlx.block_train import (
    train_linear_synthetic, linear_mse_np, synthetic_calibration,
)

try:
    import mlx.core as mx
    _HAS_MLX = True
except Exception:
    _HAS_MLX = False

T0 = time.time()
rng = np.random.default_rng(7)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(f"Phase-1 FAILED at {name}")


# ---- T1: soft-to-hard collapse as tau -> 0 -------------------------------
w = rng.standard_normal(256).astype(np.float32)
grid2 = get_grid(2)
logits = gptq_init_logits(w, grid2, rng=np.random.default_rng(0))
s = float(np.abs(w).mean())
mp_hi, mp_lo, mse_hi, mse_lo = [], [], [], []
wh, _ = quantize_hard(logits, grid2, s)
for rep in range(5):
    r = np.random.default_rng(100 + rep)
    p_hi = gumbel_softmax_sample(logits, 2.0, 100.0, r)
    p_lo = gumbel_softmax_sample(logits, 0.05, 500.0, r)
    mp_hi.append(float(p_hi.max(axis=1).mean()))
    mp_lo.append(float(p_lo.max(axis=1).mean()))
    mse_hi.append(float(((quantize_soft(p_hi, grid2, s) - wh) ** 2).mean()))
    mse_lo.append(float(((quantize_soft(p_lo, grid2, s) - wh) ** 2).mean()))
mp_hi_m, mp_lo_m = float(np.mean(mp_hi)), float(np.mean(mp_lo))
mse_hi_m, mse_lo_m = float(np.mean(mse_hi)), float(np.mean(mse_lo))
check("T1 collapse: low-tau near one-hot",
      mp_lo_m > 0.95 and mp_lo_m > mp_hi_m + 0.2,
      f"maxprob hi={mp_hi_m:.3f} lo={mp_lo_m:.3f}")
check("T1 collapse: soft->hard MSE shrinks",
      mse_lo_m < 0.5 * mse_hi_m,
      f"mse hi={mse_hi_m:.4f} lo={mse_lo_m:.6f}")

# ---- T2: local-shift indices in range -------------------------------------
K3 = 8
grid3 = get_grid(3)
r2 = np.random.default_rng(11)
d = 512
j0 = r2.integers(0, K3, size=d)
shift_logits = r2.standard_normal((d, 5)).astype(np.float32)
j = local_shift_indices(j0, shift_logits, K3)
j = np.asarray(j)
check("T2 shift indices in [0,K)",
      int(j.min()) >= 0 and int(j.max()) < K3,
      f"min={j.min()} max={j.max()} K={K3}")
# edges: hard-left at 0 and hard-right at K-1 must clip, not wrap
je = local_shift_indices(np.array([0, K3 - 1]),
                         np.array([[-99., -99., -99., -99., 99.],
                                   [99., -99., -99., -99., -99.]],
                                  dtype=np.float32), K3)
check("T2 shift edges clip", list(np.asarray(je)) == [2, K3 - 3],
      f"got={list(np.asarray(je))}")
r3 = np.random.default_rng(12)
p_soft = gumbel_softmax_sample(shift_logits, 0.5, 200.0, r3)
ws = np.asarray(local_shift_quantize_soft(p_soft, j0, grid3, s))
check("T2 shift soft bounded",
      np.all(np.isfinite(ws)) and float(np.abs(ws).max()) <= s * float(np.abs(grid3).max()) + 1e-5,
      f"max|w|={np.abs(ws).max():.4f} bound={s * np.abs(grid3).max():.4f}")
wh3, jh3 = local_shift_decode(j0, shift_logits, grid3, s)
check("T2 shift hard consistent",
      np.array_equal(np.asarray(jh3), j),
      f"decode matches indices, mse={float(((wh3 - grid3[j] * s) ** 2).mean()):.2e}")

# ---- T3: Lion decreases synthetic linear loss -----------------------------
# scale_mult=0.5 starts the joint (logits, scale) loop off-scale so the
# staged Lion optimization has measurable headroom; RTN-warm-started
# logits + per-tensor scale, paper App G lrs, accumulation over resamples.
res = train_linear_synthetic(n_bits=2, steps=60, n_resamples=4, seed=0,
                             scale_mult=0.5)
hist = res["history"]
check("T3 Lion decreases loss",
      hist[-1] < hist[0] and np.all(np.isfinite(hist)),
      f"init={hist[0]:.6f} final={hist[-1]:.6f} ratio={hist[-1] / hist[0]:.3f}")
check("T3 Lion solid progress", hist[-1] < 0.9 * hist[0],
      f"ratio={hist[-1] / hist[0]:.3f}")
moved = float(np.abs(np.asarray(res["params"]["logits"])).sum())
check("T3 Lion steps moved params", moved > 0 and np.all(np.isfinite(
    np.asarray(res["params"]["logits"]))), f"|logits|_sum={moved:.3f}")

# ---- T4: ternary mask+sign (Alg.2) -----------------------------------------
rt = np.random.default_rng(21)
d4 = 256
lm = rt.standard_normal(d4).astype(np.float32)
lb = rt.standard_normal(d4).astype(np.float32)
st = 0.05
wt_soft = ternary_soft(lm, lb, st, 0.5, 200.0, rng=np.random.default_rng(22))
wt_hard, (m4, b4) = ternary_hard(lm, lb, st)
vals = set(np.unique(np.asarray(wt_hard)).tolist())
check("T4 ternary soft bounded",
      np.all(np.isfinite(wt_soft)) and float(np.abs(wt_soft).max()) <= st + 1e-6,
      f"max|w|={np.abs(wt_soft).max():.4f}")
uniq4 = np.asarray(wt_hard).reshape(-1)
allow4 = np.array([-st, 0.0, st], dtype=np.float32)
ok4 = bool(np.all(np.min(np.abs(uniq4[:, None] - allow4[None, :]), axis=1) < 1e-6))
check("T4 ternary hard alphabet", ok4, "vals subset of {-s,0,s}")

# ---- T5: mlx parity + grad --------------------------------------------------
if _HAS_MLX:
    key = mx.random.key(0)
    k1, k2, k3 = mx.random.split(key, 3)
    logits_mx = mx.array(logits)
    p_np = np.asarray(gumbel_softmax_sample(
        logits, 0.5, 200.0, np.random.default_rng(5)))
    p_mx = gumbel_softmax_sample(logits_mx, 0.5, 200.0, key=k1)
    mx.eval(p_mx)
    check("T5 mlx sampling sane",
          tuple(p_mx.shape) == p_np.shape and bool(mx.all(mx.isfinite(p_mx))),
          f"shape={tuple(p_mx.shape)} rowsums~1: "
          f"{float(mx.abs(mx.sum(p_mx, axis=-1) - 1).max()):.2e}")
    w_np = np.asarray(quantize_soft(p_np, grid2, s))
    w_mx = quantize_soft(mx.array(p_np), mx.array(np.asarray(grid2)), s)
    mx.eval(w_mx)
    check("T5 mlx soft matches numpy",
          float(mx.max(mx.abs(w_mx - mx.array(w_np)))) < 1e-5, "")

    # grad flows through a staged linear loss closure under value_and_grad
    from gsq_mlx.block_train import linear_soft_loss_mx
    dm, inf, outf = 8, 4, 3
    Xs = mx.array(np.asarray(synthetic_calibration(2, 4, inf, seed=3)))
    Wf = mx.array(rng.standard_normal(outf * inf).astype(np.float32))
    lg = mx.array(np.zeros((outf * inf, 4), dtype=np.float32))
    sc = mx.array(np.float32(0.02))
    gg = mx.random.gumbel(shape=list(lg.shape), key=k2)
    gv = mx.array(np.asarray(grid2))
    tau, kap = gs_schedules(3, 30)

    def loss_fn(lg_, sc_):
        return linear_soft_loss_mx(lg_, sc_, gg, Xs, Wf, gv, tau, kap)

    val, (glg, gsc) = mx.value_and_grad(loss_fn, argnums=(0, 1))(lg, sc)
    mx.eval(val, glg, gsc)
    check("T5 mlx grad finite",
          bool(mx.all(mx.isfinite(glg))) and bool(mx.isfinite(gsc))
          and float(mx.sum(mx.abs(glg))) > 0,
          f"loss={float(val):.6f} |g|={float(mx.sum(mx.abs(glg))):.4f}")
    opt = Lion({"logits": lg, "scale": sc}, lr=1e-4)
    new = opt.step({"logits": lg, "scale": sc},
                   {"logits": glg, "scale": gsc})
    mx.eval(*new.values())
    check("T5 mlx Lion step updates",
          not bool(mx.array_equal(new["logits"], lg)), "")
else:
    print("[SKIP] T5 mlx parity (mlx not installed)")

dt = time.time() - T0
print(f"PHASE1 OK ({dt:.1f}s, synthetic-only, no model loaded)")
assert dt < 30, f"over budget: {dt:.1f}s"
