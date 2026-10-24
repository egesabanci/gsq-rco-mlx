"""RCO for MLX — budget manifold + DP knapsack (paper Sec 2-3, App B.1).

Dual backend: every continuous op accepts numpy arrays (fallback, keeps
``scripts/smoke_test.py`` green without mlx) AND ``mlx.core`` arrays
(real on-device path). Dispatch rule: if any input is an ``mx.array``,
the whole op runs in pure ``mx`` ops; otherwise numpy.

Grad rule (so losses are usable under ``mx.grad`` / ``mx.value_and_grad``):
``softmax`` / ``expected_cost`` / ``normal_vector`` / ``project_tangent`` /
``transport`` / ``ste_grad`` use only differentiable ops of their backend —
no RNG, no Python-side branching on array values inside the diff path.
``retract`` (binary search, Prop.3) and ``dp_knapsack`` / ``topk_2option``
(discrete decode, Eq.4) are non-differentiable projection/decode steps by
design: ``retract`` branches on evaluated scalars, the DP runs on CPU
(paper App B.1: DP on CPU with a device sync per call).

Paper mapping:
  Eq.1 C(a)=sum_i w_i <p_i,c>  -> expected_cost
  Eq.2 normal nik = w_i p_ik (c_k - E[c]) -> normal_vector
  Sec 2 project gtan = g - (<g,n>/||n||^2) n -> project_tangent
  Prop.3 binary search t* s.t. C(a + t*c) = B -> retract
  App A.5 transport m <- m - (<m,n'>/||n'||^2) n' -> transport
  Eq.4 multiple-choice knapsack, O(N*K*B') -> dp_knapsack
  Sec 3.1 / App B.1 two-option keep/prune -> topk_2option (O(N log N))
"""
from __future__ import annotations
import numpy as np

try:
    import mlx.core as mx  # type: ignore
    _HAS_MLX = True
except Exception:
    mx = None  # type: ignore
    _HAS_MLX = False


def _is_mx(x) -> bool:
    return _HAS_MLX and isinstance(x, mx.array)


def _use_mx(*xs) -> bool:
    return any(_is_mx(x) for x in xs)


def _as_mx_like(ref, arr) -> object:
    """Numpy/cost/weight constant as an mx array matching ref dtype."""
    if _is_mx(arr):
        return arr.astype(ref.dtype)
    return mx.array(np.asarray(arr)).astype(ref.dtype)


# ------------------------------------------------------------- core ops ---

def softmax(a):
    """Row-wise softmax. Pure + grad-safe in both backends."""
    if _use_mx(a):
        return mx.softmax(a, axis=1)
    a = np.asarray(a, dtype=np.float64)
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def expected_cost(logits, costs, weights):
    """Expected cost C(a) = sum_i w_i <p_i, c> (paper Eq.1).

    Numpy path returns a Python float; mx path returns a 0-d mx array
    (stays in the graph under mx.grad — do NOT float() it inside losses).
    """
    if _use_mx(logits, costs, weights):
        c = _as_mx_like(logits, costs)
        w = _as_mx_like(logits, weights)
        p = mx.softmax(logits, axis=1)
        return mx.sum(mx.expand_dims(w, 1) * p * mx.expand_dims(c, 0))
    p = softmax(np.asarray(logits))
    c = np.asarray(costs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return float((w[:, None] * p * c[None, :]).sum())


def normal_vector(logits, costs, weights):
    """Budget-manifold normal n_ik = w_i p_ik (c_k - E_{p_i}[c]) (Eq.2)."""
    if _use_mx(logits, costs, weights):
        c = _as_mx_like(logits, costs)
        w = _as_mx_like(logits, weights)
        p = mx.softmax(logits, axis=1)
        exp_c = mx.sum(p * mx.expand_dims(c, 0), axis=1, keepdims=True)
        return mx.expand_dims(w, 1) * p * (mx.expand_dims(c, 0) - exp_c)
    p = softmax(np.asarray(logits))
    c = np.asarray(costs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    exp_c = (p * c[None, :]).sum(axis=1, keepdims=True)
    return w[:, None] * p * (c[None, :] - exp_c)


def project_tangent(grad, normal):
    """Tangent projection g - (<g,n>/||n||^2) n (paper Sec 2). Pure."""
    if _use_mx(grad, normal):
        denom = mx.sum(normal * normal) + 1e-12
        return grad - (mx.sum(grad * normal) / denom) * normal
    g = np.asarray(grad, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    denom = float((n * n).sum()) + 1e-12
    return g - (float((g * n).sum()) / denom) * n


def transport(m, normal_new):
    """Vector transport by projection onto the new tangent plane
    (paper Sec 2 / App A.5). Only the first moment is transported;
    the Adam second moment has no tangent structure."""
    return project_tangent(m, normal_new)


def _feasible_range(costs, weights):
    c = np.asarray(costs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return float(w.sum() * c.min()), float(w.sum() * c.max())


def retract(logits, costs, weights, budget, lo=-50.0, hi=50.0, iters=40):
    """Binary-search retract t* s.t. C(a + t*c) = B (paper Prop.3).

    C(a + t*c) is strictly increasing in t (dC/dt = sum_i w_i Var[c] > 0
    when costs are not all equal), so bisection converges exponentially
    (~34 iters theory for R=100/eps=1e-8; 40 default). Non-differentiable
    projection step: do NOT call inside an mx.grad closure — step, then
    retract, then continue (paper Alg.1 lines 11-12).
    """
    blo, bhi = _feasible_range(costs, weights)
    if not (blo - 1e-9 <= budget <= bhi + 1e-9):
        raise ValueError(
            f"budget {budget} outside feasible [{blo}, {bhi}] "
            f"(sum w * min/max cost)")
    if _use_mx(logits, costs, weights):
        a = logits if _is_mx(logits) else mx.array(np.asarray(logits))
        c = _as_mx_like(a, costs)
        w = _as_mx_like(a, weights)
        c1 = mx.expand_dims(c, 0)
        w1 = mx.expand_dims(w, 1)
        lo_f, hi_f = float(lo), float(hi)
        for _ in range(iters):
            mid = 0.5 * (lo_f + hi_f)
            cand = a + mid * c1
            cv = mx.sum(w1 * mx.softmax(cand, axis=1) * c1)
            mx.eval(cv)
            if float(cv.item()) < float(budget):
                lo_f = mid
            else:
                hi_f = mid
        return a + (0.5 * (lo_f + hi_f)) * c1
    a = np.asarray(logits, dtype=np.float64).copy()
    c = np.asarray(costs, dtype=np.float64)
    lo_f, hi_f = float(lo), float(hi)
    for _ in range(iters):
        mid = 0.5 * (lo_f + hi_f)
        if expected_cost(a + mid * c[None, :], c,
                         np.asarray(weights, dtype=np.float64)) < budget:
            lo_f = mid
        else:
            hi_f = mid
    return a + (0.5 * (lo_f + hi_f)) * c[None, :]


# ------------------------------------------------------- discrete decode ---

def assignment_value(z: np.ndarray, scores: np.ndarray) -> float:
    """Objective value sum_{i,k} z_ik * scores_ik of a one-hot assignment."""
    return float((np.asarray(z) * np.asarray(scores)).sum())


def assignment_cost(z: np.ndarray, costs_int: np.ndarray,
                    weights_int: np.ndarray) -> int:
    """Integer cost sum_i w_i * c_{k(i)} of a one-hot assignment."""
    z = np.asarray(z)
    c = np.asarray(costs_int).reshape(-1)
    w = np.asarray(weights_int).reshape(-1)
    return int(((z * c[None, :]).sum(axis=1) * w).sum())


def topk_2option(scores: np.ndarray, costs_int: np.ndarray,
                 weights_int: np.ndarray, budget_int: int) -> np.ndarray:
    """Exact O(N log N) fast path for K=2 with uniform weights
    (paper Sec 3.1 / App B.1: keep/prune costs {0,1}, budget fixes count).

    Exact whenever all group weights are equal: every group has the same
    cost step, so the knapsack reduces to picking the n_hi best logit
    deltas. Non-uniform weights fall back to full DP (use dp_knapsack).
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError(f"topk_2option needs scores [N,2], got {s.shape}")
    N = s.shape[0]
    c = np.asarray(costs_int).reshape(-1).astype(np.int64)
    w = np.asarray(weights_int).reshape(-1).astype(np.int64)
    if c.shape != (2,) or w.shape != (N,):
        raise ValueError("costs must be [2], weights [N]")
    if int(budget_int) < 0:
        raise ValueError("budget_int must be >= 0")
    if np.any(c < 0) or np.any(w < 0):
        raise ValueError("costs/weights must be nonneg ints")
    if np.any(w != w[0]):
        raise ValueError("topk_2option is exact only for uniform weights; "
                         "use dp_knapsack for non-uniform weights")
    w0 = int(w[0])
    c0, c1 = int(c[0]), int(c[1])
    z = np.zeros((N, 2))
    if c0 == c1:  # cost-blind: per-row argmax (any feasible pick)
        z[np.arange(N), s.argmax(axis=1)] = 1.0
        return z
    lo_k, hi_k = (0, 1) if c0 < c1 else (1, 0)
    lo_c, hi_c = int(c[lo_k]), int(c[hi_k])
    base = N * w0 * lo_c
    step = w0 * (hi_c - lo_c)
    if base > int(budget_int):
        raise ValueError(
            f"infeasible: cheapest cost {base} > budget {budget_int}")
    n_hi = (int(budget_int) - base) // step
    n_hi = min(max(n_hi, 0), N)
    delta = s[:, hi_k] - s[:, lo_k]
    z[:, lo_k] = 1.0
    if n_hi > 0:
        idx = np.argsort(delta, kind="stable")[-n_hi:]
        z[idx, lo_k] = 0.0
        z[idx, hi_k] = 1.0
    return z


def dp_knapsack(scores: np.ndarray, costs_int: np.ndarray,
                weights_int: np.ndarray, budget_int: int,
                use_topk: bool = True) -> np.ndarray:
    """Multiple-choice knapsack, paper Eq.4, O(N*K*B'). Returns one-hot z.

    Each row picks exactly one k, sum_i w_i*c_{k(i)} <= B (integer grid;
    Sec 3.1: per-group costs are rounded to this grid — the source of the
    small decoded-budget deviations). CPU-side by design (App B.1);
    mx-array scores are evaluated + converted, result is numpy one-hot.

    K=2 with uniform weights auto-dispatches to the exact O(N log N)
    top-k path (disable with use_topk=False to force full DP).
    Tiny-model sizes only; uses -inf DP. Raises ValueError when the
    cheapest assignment already exceeds the budget.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] < 1 or s.shape[1] < 1:
        raise ValueError(f"scores must be [N,K], got {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    N, K = s.shape
    c = np.asarray(costs_int).reshape(-1).astype(np.int64)
    w = np.asarray(weights_int).reshape(-1).astype(np.int64)
    B = int(budget_int)
    if c.shape != (K,) or w.shape != (N,):
        raise ValueError(
            f"costs must be [{K}], weights [{N}], "
            f"got {c.shape}/{w.shape}")
    if B < 0:
        raise ValueError("budget_int must be >= 0")
    if np.any(c < 0) or np.any(w < 0):
        raise ValueError("costs/weights must be nonneg ints")
    if int((w * c.min()).sum()) > B:
        raise ValueError(
            f"infeasible: cheapest cost {int((w * c.min()).sum())} "
            f"> budget {B}")
    if K == 1:
        return np.ones((N, 1))
    if use_topk and K == 2 and np.all(w == w[0]):
        return topk_2option(s, c, w, B)

    NEG = -1e100
    dp = np.full(B + 1, NEG)
    dp[0] = 0.0
    choice = np.zeros((N, B + 1), dtype=np.int32)
    prev = np.zeros((N, B + 1), dtype=np.int32)
    step_cost = (w[:, None] * c[None, :]).astype(np.int64)  # [N,K]
    for i in range(N):
        ndp = np.full(B + 1, NEG)
        sci, stepi = s[i], step_cost[i]
        for b in range(B + 1):
            best, bk, pb = NEG, 0, 0
            for k in range(K):
                cc = int(stepi[k])
                if b - cc >= 0 and dp[b - cc] > NEG / 2:
                    v = dp[b - cc] + float(sci[k])
                    if v > best:
                        best, bk, pb = v, k, b - cc
            ndp[b], choice[i, b], prev[i, b] = best, bk, pb
        dp = ndp
    b = int(np.argmax(dp))
    z = np.zeros((N, K))
    for i in range(N - 1, -1, -1):
        k = int(choice[i, b])
        z[i, k] = 1.0
        b = int(prev[i, b])
    return z
