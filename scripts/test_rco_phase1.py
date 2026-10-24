"""Phase-1 RCO acceptance test (OWNED by RCO side).

Proves, with SYNTHETIC logits / tiny mx arrays only (no model load,
no safetensors — memory-critical Mac 17GB/~2.5GB free), all under 30s:

  T1 retract exactness: |C - B| < 1e-6 after binary-search retract
     (paper Prop.3), numpy + mx paths.
  T2 projection kills normal: <g_tan, n> ~= 0 (paper Sec 2; removes the
     constraint-normal STE bias, Prop.6), numpy + mx paths.
  T3 DP optimal: dp_knapsack matches brute-force optimum on a tiny
     N=6 K=3 random instance (paper Eq.4); top-k fast path matches
     brute force + full DP on a 2-option keep/prune instance (Sec 3.1).
  T4 10-step RCO loop holds |C - B| < 1e-6 every step (Alg.1 lines
     11-12); final DP decode respects the integer budget; tau schedule
     runs 1.0 -> 0.01 exponentially.
  T5 mlx parity + grad: mx ops match numpy; an STE-style loss closure
     differentiates under mx.value_and_grad (skipped if no mlx).

Usage: .venv/bin/python scripts/test_rco_phase1.py
Exit nonzero on any failure.
"""
from __future__ import annotations
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from rco_mlx.manifold import (
    softmax, expected_cost, normal_vector, project_tangent, transport,
    retract, dp_knapsack, topk_2option, assignment_cost, assignment_value,
)
from rco_mlx.rco import (
    tau_schedule, sample_gumbel, perturbed_logits, soft_probs, ste_grad,
    to_int_grid, synthetic_kl, assemble_weights_stub, rco_loop,
    TAU0, TAU_MIN,
)

try:
    import mlx.core as mx
    _HAS_MLX = True
except Exception:
    _HAS_MLX = False

T0 = time.time()


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(f"Phase-1 FAILED at {name}")


COSTS = np.array([2., 3., 4., 8.])
CI = np.array([2, 3, 4, 8])

# ---- T1: retract exactness ------------------------------------------------
rng = np.random.default_rng(0)
N1, B1 = 20, 2.6 * 20
w1 = np.ones(N1)
a1 = rng.standard_normal((N1, 4))  # off-manifold init
c0 = expected_cost(a1, COSTS, w1)
a1r = retract(a1, COSTS, w1, budget=B1)
c1 = expected_cost(a1r, COSTS, w1)
check("T1 retract exact |C-B|<1e-6", abs(c1 - B1) < 1e-6,
      f"init={c0:.4f} final={c1:.10f} target={B1}")
# zeros init (uniform p, C = mean cost * N) also retracts exactly
a1z = retract(np.zeros((N1, 4)), COSTS, w1, budget=B1)
check("T1 retract from uniform", abs(expected_cost(a1z, COSTS, w1) - B1) < 1e-6, "")
if _HAS_MLX:
    am = mx.array(np.asarray(a1, dtype=np.float32))
    ar = retract(am, COSTS, w1, budget=B1)
    mx.eval(ar)
    cm = expected_cost(ar, COSTS, w1)
    mx.eval(cm)
    check("T1 retract exact (mx)", abs(float(cm.item()) - B1) < 1e-4,
          f"cost={float(cm.item()):.10f} (float32: ~6e-6/ulp at C~52)")

# ---- T2: projection kills the normal component -----------------------------
g2 = rng.standard_normal((N1, 4))
n2 = normal_vector(a1r, COSTS, w1)
g2t = project_tangent(g2, n2)
res2 = abs(float((g2t * n2).sum())) / (float(np.abs(g2t).sum()) + 1e-12)
check("T2 projection kills normal", res2 < 1e-9, f"|<gtan,n>|/|gtan|_1={res2:.2e}")
# transport keeps momentum tangent at the new point
m2 = transport(rng.standard_normal((N1, 4)), normal_vector(a1r, COSTS, w1))
n2b = normal_vector(a1r, COSTS, w1)
res2b = abs(float((m2 * n2b).sum())) / (float(np.abs(m2).sum()) + 1e-12)
check("T2 transport tangent", res2b < 1e-9, f"|<m,n'>|/|m|_1={res2b:.2e}")
if _HAS_MLX:
    gm = mx.array(g2.astype(np.float32))
    nm = normal_vector(mx.array(np.asarray(a1r, dtype=np.float32)), COSTS, w1)
    gt = project_tangent(gm, nm)
    mx.eval(gt, nm)
    num = mx.sum(gt * nm)
    mx.eval(num)
    check("T2 projection kills normal (mx)", abs(float(num.item())) < 1e-4, "")

# ---- T3: DP matches brute force --------------------------------------------
rb = np.random.default_rng(3)
N3, K3 = 6, 3
scores3 = rb.standard_normal((N3, K3))
costs3 = np.array([2, 3, 5])
weights3 = np.array([1, 2, 1, 1, 2, 1])
B3 = int((weights3 * costs3.min()).sum()) + 4  # tight but feasible
best_v, best_c = -1e100, None
for combo in itertools.product(range(K3), repeat=N3):
    combo = list(combo)
    cc = int(sum(weights3[i] * costs3[k] for i, k in enumerate(combo)))
    if cc <= B3:
        vv = float(sum(scores3[i, k] for i, k in enumerate(combo)))
        if vv > best_v:
            best_v, best_c = vv, cc
z3 = dp_knapsack(scores3, costs3, weights3, B3)
v3, c3 = assignment_value(z3, scores3), assignment_cost(z3, costs3, weights3)
check("T3 DP matches brute force", abs(v3 - best_v) < 1e-9 and c3 <= B3,
      f"dp={v3:.6f} (cost {c3}) brute={best_v:.6f} (cost {best_c}) B={B3}")
# 2-option keep/prune: top-k == full DP == brute force
rk = np.random.default_rng(5)
Nk = 8
scoresk = rk.standard_normal((Nk, 2))
costsk = np.array([0, 1])
weightsk = np.ones(Nk, dtype=int)
Bk = 3  # keep 3 pruned...: exactly 3 picks of option 1
zt = topk_2option(scoresk, costsk, weightsk, Bk)
zd = dp_knapsack(scoresk, costsk, weightsk, Bk, use_topk=False)
best_k = -1e100
for combo in itertools.product(range(2), repeat=Nk):
    combo = list(combo)
    if sum(combo) <= Bk:
        vv = float(sum(scoresk[i, k] for i, k in enumerate(combo)))
        best_k = max(best_k, vv)
vt, vd = assignment_value(zt, scoresk), assignment_value(zd, scoresk)
check("T3 top-k == DP == brute force",
      abs(vt - best_k) < 1e-9 and abs(vd - best_k) < 1e-9
      and assignment_cost(zt, costsk, weightsk) <= Bk,
      f"topk={vt:.6f} dp={vd:.6f} brute={best_k:.6f}")

# ---- T4: 10-step loop holds the budget --------------------------------------
N4, B4 = 12, 3.0 * 12
w4 = np.ones(N4)
a4 = np.random.default_rng(1).standard_normal((N4, 4))
res4 = rco_loop(a4, COSTS, w4, B4, steps=10, g=4, lr=0.1, seed=0)
worst4 = max(abs(h["cost"] - B4) for h in res4["history"])
check("T4 loop holds |C-B|<1e-6 all 10 steps", worst4 < 1e-6,
      f"worst={worst4:.2e} final={res4['history'][-1]['cost']:.10f}")
check("T4 final decode within budget",
      res4["decoded_cost"] <= res4["budget_int"],
      f"decoded={res4['decoded_cost']} B'={res4['budget_int']}")
taus = [h["tau"] for h in res4["history"]]
check("T4 tau 1.0 -> 0.01 exponential",
      abs(taus[0] - 1.0) < 1e-12 and abs(taus[-1] - 0.01) < 1e-12
      and all(b <= a for a, b in zip(taus, taus[1:])),
      f"tau0={taus[0]} tau9={taus[-1]}")
check("T4 tau_schedule endpoints",
      abs(tau_schedule(0, 10) - TAU0) < 1e-12
      and abs(tau_schedule(9, 10) - TAU_MIN) < 1e-12, "")
# KL + assembly stubs behave (synthetic only)
p4 = np.asarray(soft_probs(np.asarray(res4["logits"]) / 0.5))
tgt4 = np.full_like(p4, 1.0 / p4.shape[1])
kl4 = synthetic_kl(p4, tgt4)
check("T4 synthetic KL finite", np.isfinite(kl4) and kl4 >= 0, f"kl={kl4:.4f}")
w4a = assemble_weights_stub(res4["z_final"], np.array([2., 3., 4., 8.]))
check("T4 assembly stub avg bits sane",
      2.0 - 1e-9 <= float(w4a.mean()) <= 8.0 + 1e-9,
      f"avg={float(w4a.mean()):.3f}")

# ---- T5: mlx parity + grad ---------------------------------------------------
if _HAS_MLX:
    key = mx.random.key(0)
    k1, k2 = mx.random.split(key, 2)
    am4 = mx.array(np.asarray(a4, dtype=np.float32))
    cm = np.array([2., 3., 4., 8.], dtype=np.float32)
    wm = np.ones(N4, dtype=np.float32)
    n_np = np.asarray(normal_vector(a4, COSTS, w4))
    n_mx = normal_vector(am4, mx.array(cm), mx.array(wm))
    mx.eval(n_mx)
    check("T5 mlx normal matches numpy",
          float(mx.max(mx.abs(n_mx - mx.array(n_np.astype(np.float32))))) < 1e-4, "")
    # STE closed form vs autograd through soft probs of same perturbed logits
    G = mx.random.gumbel(shape=list(am4.shape), key=k1)
    tau5 = 0.5
    h5 = perturbed_logits(am4, G, tau5)
    p5 = soft_probs(h5)
    up5 = mx.array(rng.standard_normal((N4, 4)).astype(np.float32))

    def ste_loss(a_, G_, up_):
        pp_ = mx.softmax((a_ + G_) / tau5, axis=-1)
        return mx.sum(pp_ * up_)

    val, ga = mx.value_and_grad(ste_loss, argnums=0)(am4, G, up5)
    mx.eval(val, ga)
    gs = ste_grad(up5, p5, tau5)
    mx.eval(gs)
    check("T5 mlx STE grad == autograd",
          bool(mx.all(mx.isfinite(ga)))
          and float(mx.max(mx.abs(ga - gs))) < 1e-4
          and float(mx.sum(mx.abs(ga))) > 0,
          f"loss={float(val.item()):.4f}")
    # mx loop holds the budget too
    resm = rco_loop(a4, COSTS, w4, B4, steps=10, g=2, lr=0.1, seed=0,
                    use_mx=True)
    worstm = max(abs(h["cost"] - B4) for h in resm["history"])
    check("T5 mx loop holds budget", worstm < 1e-4, f"worst={worstm:.2e} (float32)")
else:
    print("[SKIP] T5 mlx parity (mlx not installed)")

dt = time.time() - T0
print(f"PHASE1 RCO OK ({dt:.1f}s, synthetic-only, no model loaded)")
assert dt < 30, f"over budget: {dt:.1f}s"
