"""Phase-1 GSQ acceptance test (OWNED by GSQ side).

Proves, with SYNTHETIC weights / tiny mx arrays only (no model load,
no safetensors — memory-critical Mac 17GB/~2.5GB free), all under 30s:

  T1 soft-to-hard collapse: mean max-prob at tau=0.05 >> tau=2.0 and
     soft weights converge to hard weights as tau -> 0 (paper Sec 3).
  T2 local-shift indices always in [0, K-1] incl. edges; soft decode
     bounded by s*max|grid| (paper Eq.6).
  T3 GSQ optimization of a synthetic linear layer (paper App G): analytic
     gradients match finite differences; the HARD loss (argmax codes + learned
     group scales) falls clearly below the App A init and below a zero-lr
     control run with the identical tau/kappa schedule.
  T4 ternary mask+sign paths: soft in [-s,s], hard in {-s,0,s} (Alg.2).
  T5 mlx parity: same ops on mx arrays match numpy; a staged linear loss
     closure differentiates under mx.value_and_grad (skipped if no mlx).

Usage: .venv/bin/python tests/test_gsq_phase1.py
Exit nonzero on any failure.
"""

from __future__ import annotations

import os
import time
import numpy as np
from gsq_rco.gsq.gptq import init_logits_2bit_from_codes, quantize_rtn
from gsq_rco.gsq.quant import get_grid, local_shift_indices, gs_schedules
from gsq_rco.gsq.lion import LR_LOGITS
from oracles import (  # reference implementations, tests only
    Lion, gumbel_softmax_sample, quantize_soft, quantize_hard,
    ternary_soft, ternary_hard, local_shift_quantize_soft, local_shift_decode,
    linear_loss_and_grads_np, synthetic_calibration, train_linear_synthetic,
)


def test_gsq_phase1():
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
            raise AssertionError(f"Phase-1 FAILED at {name}")

    w = rng.standard_normal(256).astype(np.float32)

    grid2 = get_grid(2)

    codes1 = quantize_rtn(w.reshape(16, 16), 2, group_size=16)["codes"].reshape(-1)

    logits = init_logits_2bit_from_codes(codes1, len(grid2), seed=0)  # App A init

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

    rg = np.random.default_rng(3)

    Wg = (rg.standard_normal((4, 8)) * 0.02).astype(np.float32)

    Xg = synthetic_calibration(4, 4, 8, seed=4)

    rtn_g = quantize_rtn(Wg, 2, group_size=4)

    logit_g = init_logits_2bit_from_codes(rtn_g["codes"], len(grid2), seed=5).astype(np.float64)

    logit_g += rg.standard_normal(logit_g.shape) * 0.02  # move off the init so every entry matters

    scales_g = rtn_g["scales"].astype(np.float64)

    ent = [(0, 0), (5, 2), (9, 1), (13, 3), (31, 0)]

    for tau_g, kappa_g in [(2.0, 100.0), (0.5, 300.0)]:
        def loss_of(lg, sc):
            return linear_loss_and_grads_np(Xg, Wg, lg, grid2, sc, 4, tau_g, kappa_g,
                                            n_resamples=1, seed=7)[0]
        _, glog, gsc = linear_loss_and_grads_np(Xg, Wg, logit_g, grid2, scales_g, 4,
                                                tau_g, kappa_g, n_resamples=1, seed=7)
        eps = 1e-6
        fd_l = []
        for dd, kk in ent:
            lp, lm = logit_g.copy(), logit_g.copy()
            lp[dd, kk] += eps
            lm[dd, kk] -= eps
            fd_l.append((loss_of(lp, scales_g) - loss_of(lm, scales_g)) / (2 * eps))
        err_l = float(np.max(np.abs(np.array(fd_l) - np.array([glog[dd, kk] for dd, kk in ent]))))
        err_l /= float(np.abs(glog).max())
        fd_s = []
        for rr, cc in [(0, 0), (1, 1), (3, 0)]:
            sp, sm = scales_g.copy(), scales_g.copy()
            sp[rr, cc] += eps
            sm[rr, cc] -= eps
            fd_s.append((loss_of(logit_g, sp) - loss_of(logit_g, sm)) / (2 * eps))
        err_s = float(np.max(np.abs(np.array(fd_s) - np.array([gsc[rr, cc] for rr, cc in [(0, 0), (1, 1), (3, 0)]]))))
        err_s /= float(np.abs(gsc).max())
        check(f"T3 analytic logit gradient matches finite differences (tau={tau_g})",
              err_l < 1e-4, f"max err / max|grad| = {err_l:.2e}")
        check(f"T3 analytic scale gradient matches finite differences (tau={tau_g})",
              err_s < 1e-6, f"max err / max|grad| = {err_s:.2e}")

    res = train_linear_synthetic(n_bits=2, seed=0)

    ctrl = train_linear_synthetic(n_bits=2, seed=0, lr_logits=0.0, lr_scales=0.0)

    logit_only = train_linear_synthetic(n_bits=2, seed=0, lr_scales=0.0)

    ratio = res["hard_final"] / res["hard_init"]

    check("T3 GSQ moves codes across the App A gap",
          res["n_flips"] > 0 and res["max_logit_move"] >= 0.03,
          f"flips={res['n_flips']} max|dlogit|={res['max_logit_move']:.3f} lr*steps={LR_LOGITS * 1280:.3f}")

    check("T3 hard loss clearly below init (>= 20% lower)",
          res["hard_final"] <= 0.8 * res["hard_init"],
          f"init={res['hard_init']:.4e} final={res['hard_final']:.4e} ratio={ratio:.3f}")

    check("T3 hard loss clearly below zero-lr control (>= 20% lower)",
          res["hard_final"] <= 0.8 * ctrl["hard_final"],
          f"control={ctrl['hard_final']:.4e} ratio={res['hard_final'] / ctrl['hard_final']:.3f}")

    check("T3 logits alone (scales frozen) clearly below init",
          logit_only["hard_final"] <= 0.8 * logit_only["hard_init"],
          f"ratio={logit_only['hard_final'] / logit_only['hard_init']:.3f} flips={logit_only['n_flips']}")

    check("T3 zero-lr control leaves the hard loss unchanged",
          ctrl["hard_final"] == ctrl["hard_init"], "")

    hs = res["soft_history"]

    print(f"      soft-loss curve (diagnostic only, not evidence): {hs[0]:.4e} -> {hs[-1]:.4e}")

    rt = np.random.default_rng(21)

    d4 = 256

    lm = rt.standard_normal(d4).astype(np.float32)

    lb = rt.standard_normal(d4).astype(np.float32)

    st = 0.05

    wt_soft = ternary_soft(lm, lb, st, 0.5, 200.0, rng=np.random.default_rng(22))

    wt_hard, (m4, b4) = ternary_hard(lm, lb, st)

    uniq4 = np.asarray(wt_hard).reshape(-1)

    allow4 = np.array([-st, 0.0, st], dtype=np.float32)

    ok4 = bool(np.all(np.min(np.abs(uniq4[:, None] - allow4[None, :]), axis=1) < 1e-6))

    check("T4 ternary soft bounded",
          np.all(np.isfinite(wt_soft)) and float(np.abs(wt_soft).max()) <= st + 1e-6,
          f"max|w|={np.abs(wt_soft).max():.4f}")

    check("T4 ternary hard alphabet", ok4, "vals subset of {-s,0,s}")

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
        from oracles import linear_soft_loss_mx
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

        val, (glg, gsc_mx) = mx.value_and_grad(loss_fn, argnums=(0, 1))(lg, sc)
        mx.eval(val, glg, gsc_mx)
        check("T5 mlx grad finite",
              bool(mx.all(mx.isfinite(glg))) and bool(mx.isfinite(gsc_mx))
              and float(mx.sum(mx.abs(glg))) > 0,
              f"loss={float(val):.6f} |g|={float(mx.sum(mx.abs(glg))):.4f}")
        opt = Lion({"logits": lg, "scale": sc}, lr=LR_LOGITS)
        new = opt.step({"logits": lg, "scale": sc},
                       {"logits": glg, "scale": gsc_mx})
        mx.eval(*new.values())
        check("T5 mlx Lion step updates",
              not bool(mx.array_equal(new["logits"], lg)), "")
    else:
        print("[SKIP] T5 mlx parity (mlx not installed)")

    dt = time.time() - T0

    print(f"PHASE1 OK ({dt:.1f}s, synthetic-only, no model loaded)")

    assert dt < 30, f"over budget: {dt:.1f}s"
