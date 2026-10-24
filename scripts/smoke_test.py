"""Smoke test with assertions (no model needed), under 5 s.

GSQ: 2-bit App A logits around signed-RTN codes collapse to hard codes as tau -> 0;
quantizer round trip. RCO: retraction hits the budget, tangent step is orthogonal to
the normal, multiple-choice DP is feasible and optimal on a small instance.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from gsq_mlx.gptq import (dequantize, init_layer, init_logits_2bit_from_codes,
                          quantize_rtn)
from gsq_mlx.quant import get_grid, gumbel_softmax_sample
from rco_mlx.manifold import (RETRACT_TOL, dp_knapsack, expected_cost,
                              normal_vector, project_tangent, retract)


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(f"smoke_test FAILED at {name}")


rng = np.random.default_rng(0)

# --- GSQ: App A logits around signed-RTN codes; soft -> hard as tau -> 0 ---
W = (rng.standard_normal((16, 64)) * 0.02).astype(np.float32)
grid = get_grid(2)
rtn = quantize_rtn(W, 2, group_size=64)
logits = init_logits_2bit_from_codes(rtn["codes"], len(grid), seed=0)
s_el = np.repeat(rtn["scales"].astype(np.float64), 64, axis=1)
hard = grid[np.argmax(logits, axis=1)].reshape(W.shape) * s_el
p_lo = gumbel_softmax_sample(logits, 0.05, 500.0, rng=np.random.default_rng(1))
p_hi = gumbel_softmax_sample(logits, 2.0, 100.0, rng=np.random.default_rng(1))
soft_lo = (p_lo @ grid).reshape(W.shape) * s_el
soft_hi = (p_hi @ grid).reshape(W.shape) * s_el
maxp_lo, maxp_hi = float(p_lo.max(axis=1).mean()), float(p_hi.max(axis=1).mean())
dev_lo = float(np.abs(soft_lo - hard).mean())
dev_hi = float(np.abs(soft_hi - hard).mean())
check("GSQ soft->hard: low-tau probabilities near one-hot",
      maxp_lo > 0.95 and maxp_lo > maxp_hi,
      f"maxprob hi={maxp_hi:.3f} lo={maxp_lo:.3f}")
check("GSQ soft->hard: soft weights converge to hard weights",
      dev_lo < 0.1 * dev_hi, f"mean|soft-hard| hi={dev_hi:.2e} lo={dev_lo:.2e}")

# --- quantizer round trip ---
wq = dequantize(rtn["codes"], rtn["scales"], 2, 64)
check("signed-RTN round trip: dequantize(codes, scales) == w_q",
      np.array_equal(wq, rtn["w_q"]), "")
check("signed-RTN round trip: |W - w_q| <= |group scale|",
      bool(np.all(np.abs(W - wq) <= np.abs(s_el) + 1e-7)),
      f"max|err|={np.abs(W - wq).max():.4f} max|s|={np.abs(s_el).max():.4f}")
check("signed-RTN codes index the 2-bit grid", int(rtn["codes"].max()) < len(grid), "")

# App A wiring: init logits are centred on the RTN codes; shapes per bit-width.
agree = float(np.mean(np.argmax(logits, axis=1) == rtn["codes"].reshape(-1)))
check("App A init: argmax of logits matches RTN codes (noise flips < 5%)",
      agree > 0.95, f"agreement={agree:.4f}")
il = {b: init_layer(b, (16, 64), 64, seed=0, w=W) for b in (1, 2, 3)}  # quantize_qwen keys
check("init_layer keys and shapes per bit-width",
      il[1]["mask_logits"].shape == (1024,) and il[2]["logits"].shape == (1024, 4)
      and il[3]["shift_logits"].shape == (1024, 5)
      and all(il[b]["w_q"].shape == W.shape and il[b]["scales"].shape == (16, 1) for b in il),
      "ternary mask/sign, 2-bit full grid, 3-bit local shift")

# --- RCO: retraction residual, tangent projection ---
N, K = 20, 4
costs = np.array([2.0, 3.0, 4.0, 8.0])
weights = np.ones(N)
budget = 2.6 * N
a = retract(rng.standard_normal((N, K)), costs, weights, budget)
resid = abs(expected_cost(a, costs, weights) - budget)
check("RCO retraction: |C(a) - B| within RETRACT_TOL",
      resid <= RETRACT_TOL, f"|C-B|={resid:.2e} tol={RETRACT_TOL:.0e}")

n_vec = normal_vector(a, costs, weights)
gt = project_tangent(rng.standard_normal((N, K)), n_vec)
cos_gn = abs(float((gt * n_vec).sum())) / (np.linalg.norm(gt) * np.linalg.norm(n_vec))
check("RCO tangent projection is orthogonal to the normal", cos_gn < 1e-8,
      f"|cos|={cos_gn:.2e}")
a2 = retract(a - 0.05 * gt, costs, weights, budget)
resid2 = abs(expected_cost(a2, costs, weights) - budget)
check("RCO retraction after a tangent step: |C - B| within RETRACT_TOL",
      resid2 <= RETRACT_TOL, f"|C-B|={resid2:.2e}")

# --- RCO: multiple-choice DP feasibility and optimality ---
costs_int = np.array([2, 3, 4, 8])
z = dp_knapsack(a, costs_int, np.ones(N, dtype=int), int(budget))
check("DP output is one-hot per group",
      bool(np.all(np.isin(z, [0, 1]))) and bool(np.all(z.sum(axis=1) == 1)), "")
spent = int(round(float((z * costs_int[None, :]).sum())))
check("DP feasibility: spent <= budget", spent <= int(budget),
      f"spent={spent} budget={int(budget)}")

Ns, Bs = 6, 20
sc = rng.standard_normal((Ns, K))
best = max(sum(sc[i, k] for i, k in enumerate(ch))
           for ch in itertools.product(range(K), repeat=Ns)
           if sum(costs_int[k] for k in ch) <= Bs)
z_small = dp_knapsack(sc, costs_int, np.ones(Ns, dtype=int), Bs)
val_dp = float((z_small * sc).sum())
check("DP optimality vs brute force (6 groups x 4 options)",
      abs(val_dp - best) < 1e-9, f"dp={val_dp:.6f} brute={best:.6f}")

print("SMOKE OK")
