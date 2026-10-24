"""Phase-1 RCO acceptance test (OWNED by RCO side).

Proves, with SYNTHETIC logits / tiny mx arrays only (no model load,
no safetensors — memory-critical Mac 17GB/~2.5GB free), all under 30s:

  T1 retract exactness: |C - B| <= 1e-8 (float64, paper App A.2 eps) and
     <= 1e-5 max(1,B) on the float32 mx path; bracket expansion for
     large-magnitude logits (|t| > 50); boundary budgets; an infeasible
     budget and the iteration cap both raise.
  T2 projection kills normal (paper Sec 2; Prop.6), numpy + mx paths.
  T3 DP: dp_knapsack == brute force on 60 random instances (negative
     scores, non-uniform weights) and == an independent reference DP on a
     mid-size instance; K=2 `<= B` top-k == DP == brute force (300 seeds,
     negative deltas); exact_count=True == brute force with exactly n
     upgrades; N=168, K=6, B'=20000 DP in < 1 s.
  T4 rco_loop contract: grad_fn required; (dL/dy, loss) recorded per step;
     callback; 10-step loop holds the integer budget; tau 1.0 -> 0.01; DP
     decode invariant to positive scaling of the scores.
  T5 mlx parity + grad: mx ops match numpy; STE closed form == autograd;
     mx loop holds the budget (skipped if no mlx).
  T6 optimizer optimizes: on a linear MCKP (DP optimum known), RCO with the
     matching gradient gets within 1% of the optimum in 100 steps on 4
     instances; the lr=0 control stays more than 1% away.
  T7 exact integer weights: integer_weights on Qwen2.5-0.5B linear sizes
     (unit 114688, weights 7/1/1/7/38/38/38); to_int_grid refuses
     fractional weights unless allow_rounding=True.

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
    expected_cost, normal_vector, project_tangent, transport,
    retract, dp_knapsack, topk_2option, assignment_cost, assignment_value,
)
from rco_mlx.rco import (
    tau_schedule, perturbed_logits, soft_probs, ste_grad,
    to_int_grid, integer_weights, synthetic_kl, assemble_weights_stub,
    rco_loop, TAU0, TAU_MIN,
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


def raises(exc, fn) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def error_text(exc, fn) -> str:
    try:
        fn()
    except exc as e:
        return str(e)
    return ""


def brute_force(s, c, w, B, count_opt=None, count=None):
    """Exhaustive max of sum z*s over assignments with sum_i w_i c_k(i) <= B
    (and, if count_opt is given, exactly `count` groups on that option)."""
    N, K = s.shape
    combos = np.array(list(itertools.product(range(K), repeat=N)), dtype=np.int64)
    vals = s[np.arange(N)[None, :], combos].sum(axis=1)
    cost = (w[None, :] * c[combos]).sum(axis=1)
    ok = cost <= B
    if count_opt is not None:
        ok &= (combos == count_opt).sum(axis=1) == count
    return float(vals[ok].max()) if ok.any() else -np.inf


def reference_dp_value(s, c, w, B):
    """Independent exact DP over reachable total costs (pure Python)."""
    best = {0: 0.0}
    for i in range(s.shape[0]):
        nxt = {}
        for cost0, val in best.items():
            for k in range(s.shape[1]):
                cc = cost0 + int(w[i]) * int(c[k])
                if cc <= B:
                    v = val + float(s[i, k])
                    if cc not in nxt or v > nxt[cc]:
                        nxt[cc] = v
        best = nxt
    return max(best.values())


COSTS = np.array([2., 3., 4., 8.])

# ---- T1: retract exactness and bracket -------------------------------------
rng = np.random.default_rng(0)
N1, B1 = 20, 2.6 * 20
w1 = np.ones(N1)
a1 = rng.standard_normal((N1, 4))  # off-manifold init
c0 = expected_cost(a1, COSTS, w1)
a1r = retract(a1, COSTS, w1, budget=B1)
c1 = expected_cost(a1r, COSTS, w1)
check("T1 retract exact |C-B|<=1e-8", abs(c1 - B1) <= 1e-8,
      f"init={c0:.4f} final={c1:.10f} target={B1}")
a1z = retract(np.zeros((N1, 4)), COSTS, w1, budget=B1)
check("T1 retract from uniform", abs(expected_cost(a1z, COSTS, w1) - B1) <= 1e-8, "")
# Root outside the paper's [-50, 50] bracket and not saturated: closer costs
# (2.0..2.3) with logits of scale 30 give t ~ -73 and a continuous residual.
C_close = np.array([2.0, 2.1, 2.2, 2.3])
B_close = 2.15 * N1
a1big = np.random.default_rng(0).standard_normal((N1, 4)) * 30.0
a1bigr = retract(a1big, C_close, w1, budget=B_close)
t_big = float(np.mean((a1bigr - a1big) / C_close[None, :]))
res_big = abs(expected_cost(a1bigr, C_close, w1) - B_close)
check("T1 bracket expansion (|t|>50, non-saturated)",
      res_big <= 1e-8 and abs(t_big) > 50, f"t={t_big:.2f} residual={res_big:.2e}")
check("T1 min-cost boundary budget",
      abs(expected_cost(retract(a1, COSTS, w1, budget=40.0), COSTS, w1) - 40.0) <= 1e-8, "")
check("T1 infeasible budget raises ValueError",
      raises(ValueError, lambda: retract(a1, COSTS, w1, budget=160.5)), "")
check("T1 iteration cap raises RuntimeError",
      raises(RuntimeError, lambda: retract(a1, COSTS, w1, budget=B1, iters=3)), "")
if _HAS_MLX:
    am = mx.array(np.asarray(a1, dtype=np.float32))
    ar = retract(am, COSTS, w1, budget=B1)
    mx.eval(ar)
    cm = expected_cost(ar, COSTS, w1)
    mx.eval(cm)
    tol_f32 = 1e-5 * max(1.0, B1)
    check("T1 retract exact (mx float32)",
          abs(float(cm.item()) - B1) <= tol_f32,
          f"cost={float(cm.item()):.10f} tol={tol_f32:.1e}")
    amb = mx.array(a1big.astype(np.float32))
    arb = retract(amb, C_close, w1, budget=B_close)
    mx.eval(arb)
    cmb = expected_cost(arb, C_close, w1)
    mx.eval(cmb)
    check("T1 bracket expansion (mx float32)",
          abs(float(cmb.item()) - B_close) <= 1e-5 * max(1.0, B_close),
          f"cost={float(cmb.item()):.6f}")

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

# ---- T3: DP matches brute force, top-k, speed ------------------------------
rb = np.random.default_rng(11)
n_ok = 0
for _ in range(60):
    N3, K3 = int(rb.integers(3, 8)), int(rb.integers(2, 5))
    s3 = rb.standard_normal((N3, K3))  # negative scores included
    c3 = rb.integers(1, 8, size=K3)
    w3 = rb.integers(1, 4, size=N3)  # non-uniform weights
    lo3 = int((w3 * c3.min()).sum())
    hi3 = int((w3 * c3.max()).sum())
    B3 = int(rb.integers(lo3, hi3 + 1))
    z3 = dp_knapsack(s3, c3, w3, B3)
    n_ok += (abs(assignment_value(z3, s3) - brute_force(s3, c3, w3, B3)) < 1e-9
             and assignment_cost(z3, c3, w3) <= B3 and np.all(z3.sum(1) == 1))
check("T3 DP == brute force (60 random instances)", n_ok == 60, f"{n_ok}/60")
z_tie = dp_knapsack(np.zeros((6, 3)), np.array([2, 3, 5]), np.ones(6, dtype=int), 20)
check("T3 zero scores pick the cheapest optimum (deterministic)",
      np.all(z_tie[:, 0] == 1) and np.array_equal(
          z_tie, dp_knapsack(np.zeros((6, 3)), np.array([2, 3, 5]), np.ones(6, dtype=int), 20)),
      "")
check("T3 infeasible budget raises ValueError",
      raises(ValueError, lambda: dp_knapsack(np.zeros((3, 2)), np.array([2, 3]),
                                             np.ones(3, dtype=int), 5)), "")
check("T3 non-integer costs are refused, not rounded",
      raises(ValueError, lambda: dp_knapsack(np.zeros((3, 2)), np.array([2.5, 3.0]),
                                             np.ones(3, dtype=int), 9)), "")

# K=2 keep/prune: `<= B` top-k == DP == brute force, negative deltas
rk = np.random.default_rng(5)
n_topk = n_z_eq = n_forced_worse = n_exact_ok = n_exact = 0
for _ in range(300):
    Nk = int(rk.integers(4, 10))
    sk = rk.standard_normal((Nk, 2))
    ck = rk.integers(0, 6, size=2)
    wk0 = int(rk.integers(1, 4))
    wk = np.full(Nk, wk0)
    base = Nk * wk0 * int(ck.min())
    Bk = int(rk.integers(base, Nk * wk0 * int(ck.max()) + 1))
    zt = topk_2option(sk, ck, wk, Bk)
    zd = dp_knapsack(sk, ck, wk, Bk, use_topk=False)
    bk = brute_force(sk, ck, wk, Bk)
    n_topk += (abs(assignment_value(zt, sk) - bk) < 1e-9
               and abs(assignment_value(zd, sk) - bk) < 1e-9
               and assignment_cost(zt, ck, wk) <= Bk)
    n_z_eq += bool(np.array_equal(zt, zd))  # generic scores: unique optimum
    if ck[0] != ck[1]:
        lo_k = 0 if ck[0] < ck[1] else 1
        hi_k = 1 - lo_k
        n_up = min((Bk - base) // (wk0 * int(ck[hi_k] - ck[lo_k])), Nk)
        ze = topk_2option(sk, ck, wk, Bk, exact_count=True)
        be = brute_force(sk, ck, wk, Bk, count_opt=hi_k, count=n_up)
        n_exact += 1
        n_exact_ok += (int(ze[:, hi_k].sum()) == n_up
                       and abs(assignment_value(ze, sk) - be) < 1e-9)
        n_forced_worse += assignment_value(ze, sk) < assignment_value(zt, sk) - 1e-9
check("T3 K=2 `<=B` top-k == DP == brute force (300 seeds)", n_topk == 300,
      f"{n_topk}/300")
check("T3 K=2 top-k z == DP z on generic scores", n_z_eq == 300, f"{n_z_eq}/300")
check("T3 exact_count == brute force with exactly n upgrades",
      n_exact_ok == n_exact and n_exact > 200,
      f"{n_exact_ok}/{n_exact}; exact count is worse than `<=B` on {n_forced_worse}")
check("T3 exact_count needs a nonzero cost step",
      raises(ValueError, lambda: topk_2option(np.zeros((3, 2)), np.array([2, 2]),
                                              np.ones(3, dtype=int), 6, exact_count=True)), "")

# Large instance: N=168, K=6, B'=20000 (binding), speed and correctness
rl = np.random.default_rng(21)
Nl, Kl, Bl = 168, 6, 20000
sl = rl.standard_normal((Nl, Kl))
cl = np.array([1, 2, 3, 4, 6, 8])
wl = rl.integers(1, 40, size=Nl)
t_dp = time.perf_counter()
zl = dp_knapsack(sl, cl, wl, Bl)
dt_dp = time.perf_counter() - t_dp
check("T3 DP N=168 K=6 B'=20000 under 1 s",
      dt_dp < 1.0 and assignment_cost(zl, cl, wl) <= Bl,
      f"{dt_dp * 1e3:.0f} ms, cost={assignment_cost(zl, cl, wl)}/{Bl}")
rm = np.random.default_rng(22)
n_ref = 0
for _ in range(3):
    Nm = 30
    sm = rm.standard_normal((Nm, 4))
    cm4 = np.array([1, 2, 4, 7])
    wm = rm.integers(1, 6, size=Nm)
    Bm = 300
    zm = dp_knapsack(sm, cm4, wm, Bm)
    n_ref += abs(assignment_value(zm, sm) - reference_dp_value(sm, cm4, wm, Bm)) < 1e-9
check("T3 DP == reference DP on mid-size instances", n_ref == 3, f"{n_ref}/3")

# ---- T4: rco_loop contract and 10-step budget hold -------------------------
N4, B4 = 12, 3.0 * 12
w4 = np.ones(N4)
r4 = np.random.default_rng(1)
a4 = r4.standard_normal((N4, 4))
s4 = r4.standard_normal((N4, 4))


def lin_grad(z, p, step, sample, rng):
    """dL/dz for L = -sum z*s: the linear objective's matching gradient."""
    return -s4


check("T4 rco_loop requires grad_fn",
      raises(ValueError, lambda: rco_loop(a4, COSTS, w4, B4, steps=2)), "")
res4 = rco_loop(a4, COSTS, w4, B4, steps=10, g=4, lr=0.1, seed=0, grad_fn=lin_grad)
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
check("T4 no loss key when grad_fn returns an array",
      "loss" not in res4["history"][0], "")
res_l = rco_loop(a4, COSTS, w4, B4, steps=3, g=4, seed=0,
                 grad_fn=lambda z, p, t, j, rng: (-s4, float(j + 1)))
check("T4 per-step mean loss recorded",
      len(res_l["history"]) == 3
      and all(abs(h["loss"] - 2.5) < 1e-12 for h in res_l["history"]),
      f"loss={res_l['history'][0]['loss']} (mean of 1..4)")
check("T4 mixed loss/array returns raise",
      raises(ValueError, lambda: rco_loop(
          a4, COSTS, w4, B4, steps=2, g=2, seed=0,
          grad_fn=lambda z, p, t, j, rng: (-s4, 1.0) if j == 0 else -s4)), "")
seen = []
rco_loop(a4, COSTS, w4, B4, steps=4, g=2, seed=0, grad_fn=lin_grad,
         callback=lambda t, lg, rec: seen.append((t, rec["t"])))
check("T4 callback runs once per step", [x for x, _ in seen] == [0, 1, 2, 3]
      and all(x == y for x, y in seen), "")
rs = np.random.default_rng(31)
n_scale = 0
for _ in range(30):
    hs = rs.standard_normal((9, 3))
    ws = rs.integers(1, 3, size=9)
    cs = np.array([1, 2, 4])
    Bs = int(ws.sum() * 2)
    n_scale += bool(np.array_equal(dp_knapsack(hs, cs, ws, Bs),
                                   dp_knapsack(hs / 0.01, cs, ws, Bs)))
check("T4 DP decode invariant to positive scaling (no double 1/tau)",
      n_scale == 30, f"{n_scale}/30")
# KL + assembly stubs behave (synthetic only)
p4 = np.asarray(soft_probs(np.asarray(res4["logits"]) / 0.5))
tgt4 = np.full_like(p4, 1.0 / p4.shape[1])
kl4 = synthetic_kl(p4, tgt4)
check("T4 synthetic KL finite", np.isfinite(kl4) and kl4 >= 0, f"kl={kl4:.4f}")
w4a = assemble_weights_stub(res4["z_final"], np.array([2., 3., 4., 8.]))
check("T4 assembly stub avg bits sane",
      2.0 - 1e-9 <= float(w4a.mean()) <= 8.0 + 1e-9,
      f"avg={float(w4a.mean()):.3f}")

# ---- T5: mlx parity + grad -------------------------------------------------
if _HAS_MLX:
    key = mx.random.key(0)
    k1 = mx.random.split(key, 2)[0]
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
    resm = rco_loop(a4, COSTS, w4, B4, steps=10, g=2, lr=0.1, seed=0,
                    use_mx=True, grad_fn=lin_grad)
    worstm = max(abs(h["cost"] - B4) for h in resm["history"])
    tol_f32 = 1e-5 * max(1.0, B4)
    check("T5 mx loop holds budget", worstm <= tol_f32,
          f"worst={worstm:.2e} (float32, tol {tol_f32:.1e})")
else:
    print("[SKIP] T5 mlx parity (mlx not installed)")

# ---- T6: the optimizer actually optimizes ----------------------------------
def make_grad(s):
    return lambda z, p, t, j, rng: -s


gaps_rco, gaps_lr0 = [], []
for seed in range(4):
    rq = np.random.default_rng(100 + seed)
    Nq, Bq = 24, 3.0 * 24
    wq = np.ones(Nq)
    sq = 1.0 + rq.random((Nq, 4))  # linear, separable value
    ci_q, wi_q, Bi_q = to_int_grid(COSTS, wq, Bq)
    vopt = assignment_value(dp_knapsack(sq, ci_q, wi_q, Bi_q), sq)
    aq = rq.standard_normal((Nq, 4)) * 0.5
    z_rco = rco_loop(aq, COSTS, wq, Bq, steps=100, g=4, lr=0.1, seed=seed,
                     grad_fn=make_grad(sq))["z_final"]
    z_lr0 = rco_loop(aq, COSTS, wq, Bq, steps=100, g=4, lr=0.0, seed=seed,
                     grad_fn=make_grad(sq))["z_final"]
    gaps_rco.append(1.0 - assignment_value(z_rco, sq) / vopt)
    gaps_lr0.append(1.0 - assignment_value(z_lr0, sq) / vopt)
check("T6 RCO within 1% of DP optimum (4 instances, 100 steps)",
      max(gaps_rco) < 0.01,
      "gaps=" + " ".join(f"{100 * x:.2f}%" for x in gaps_rco))
check("T6 lr=0 control stays more than 1% away",
      min(gaps_lr0) > 0.01,
      "gaps=" + " ".join(f"{100 * x:.2f}%" for x in gaps_lr0))

# ---- T7: exact integer weights and the refusal to round ---------------------
qwen_counts = np.array([802816, 114688, 114688, 802816,
                        4358144, 4358144, 4358144])  # q k v o gate up down
w_q, unit_q = integer_weights(qwen_counts)
check("T7 Qwen unit 114688 and weights 7/1/1/7/38/38/38",
      unit_q == 114688 and w_q.tolist() == [7, 1, 1, 7, 38, 38, 38]
      and np.array_equal(w_q * unit_q, qwen_counts),
      f"unit={unit_q} w={w_q.tolist()}")
msg = error_text(ValueError, lambda: to_int_grid(COSTS, qwen_counts / 1e6, 52.0))
check("T7 to_int_grid refuses fractional weights",
      "integer_weights" in msg, f"msg='{msg[:60]}...'")
ci7, wi7, Bi7 = to_int_grid(COSTS, np.array([1.0, 2.0]), 4.0)
check("T7 integer weights pass unchanged", wi7.tolist() == [1, 2], "")
ci7r, wi7r, _ = to_int_grid(COSTS, np.array([0.4, 2.6]), 4.0, allow_rounding=True)
check("T7 allow_rounding=True rounds weights",
      wi7r.tolist() == [0, 3], f"w={wi7r.tolist()}")
check("T7 integer_weights refuses fractional and all-zero counts",
      raises(ValueError, lambda: integer_weights([1.5, 2.0]))
      and raises(ValueError, lambda: integer_weights([0, 0])), "")

dt = time.time() - T0
print(f"PHASE1 RCO OK ({dt:.1f}s, synthetic-only, no model loaded)")
assert dt < 30, f"over budget: {dt:.1f}s"
