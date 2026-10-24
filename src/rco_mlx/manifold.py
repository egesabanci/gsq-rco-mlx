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

# Retraction tolerances (App A.2). The float64 path uses the paper's
# eps = 1e-8 on the cost residual; the float32 mx path cannot resolve the
# residual below ~1e-6 relative, so it uses a tolerance relative to |B|.
RETRACT_TOL = 1e-8
RETRACT_TOL_F32 = 1e-5
RETRACT_MAX_SHIFT = 1e6


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


def _bracket_root(cost_at, budget: float, eps: float, lo: float, hi: float,
                  max_shift: float):
    """Widen [lo, hi] geometrically until cost_at(lo) <= B <= cost_at(hi)
    (within eps). cost_at is nondecreasing in t (Prop.3), so the root lies
    inside the returned bracket. Raises ValueError past |t| > max_shift."""
    span = max(hi - lo, 1.0)
    while cost_at(lo) > budget + eps:
        hi, lo, span = lo, lo - 2.0 * span, 2.0 * span
        if abs(lo) > max_shift:
            raise ValueError(
                f"budget {budget} not bracketed for |t| <= {max_shift:g}")
    while cost_at(hi) < budget - eps:
        lo, hi, span = hi, hi + 2.0 * span, 2.0 * span
        if abs(hi) > max_shift:
            raise ValueError(
                f"budget {budget} not bracketed for |t| <= {max_shift:g}")
    return lo, hi


def retract(logits, costs, weights, budget, lo=-50.0, hi=50.0, iters=200,
            tol=None, max_shift=RETRACT_MAX_SHIFT):
    """Retraction t* s.t. C(a + t*c) = B (paper Prop.3, App A.2).

    C(a + t*c) is strictly increasing in t unless all costs are equal
    (dC/dt = sum_i w_i Var_{p_i}[c] > 0), so the root is unique on the
    feasible range. The bracket [lo, hi] is verified and widened
    geometrically if it misses the root (up to |t| <= max_shift). Bisection
    stops once |C - B| <= tol: RETRACT_TOL (paper eps) on the float64 path,
    RETRACT_TOL_F32 * max(1, |B|) on the mx float32 path. Raises RuntimeError
    if `iters` bisection steps do not reach tol. Non-differentiable
    projection step: do NOT call inside an mx.grad closure.
    """
    blo, bhi = _feasible_range(costs, weights)
    if not (blo - 1e-9 <= budget <= bhi + 1e-9):
        raise ValueError(
            f"budget {budget} outside feasible [{blo}, {bhi}] "
            f"(sum w * min/max cost)")
    if _use_mx(logits, costs, weights):
        a = logits if _is_mx(logits) else mx.array(np.asarray(logits))
        c1 = mx.expand_dims(_as_mx_like(a, costs), 0)
        w1 = mx.expand_dims(_as_mx_like(a, weights), 1)
        eps = (RETRACT_TOL_F32 * max(1.0, abs(float(budget)))
               if tol is None else float(tol))

        def shifted(t):
            return a + t * c1

        def cost_at(t):
            cv = mx.sum(w1 * mx.softmax(shifted(t), axis=1) * c1)
            mx.eval(cv)
            return float(cv.item())
    else:
        a = np.asarray(logits, dtype=np.float64)
        c = np.asarray(costs, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        eps = RETRACT_TOL if tol is None else float(tol)

        def shifted(t):
            return a + t * c[None, :]

        def cost_at(t):
            return expected_cost(shifted(t), c, w)

    lo_f, hi_f = _bracket_root(cost_at, float(budget), eps, float(lo),
                               float(hi), float(max_shift))
    res = float("inf")
    for _ in range(iters):
        mid = 0.5 * (lo_f + hi_f)
        if not lo_f < mid < hi_f:  # bracket exhausted at float resolution
            break
        cm = cost_at(mid)
        res = abs(cm - float(budget))
        if res <= eps:
            return shifted(mid)
        if cm < budget:
            lo_f = mid
        else:
            hi_f = mid
    raise RuntimeError(
        f"retract: no shift reached |C-B| <= {eps:.3g} (best {res:.3g}) "
        f"within {iters} steps or float resolution")


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


def _int_array(x, name: str) -> np.ndarray:
    """Flat int64 array. Floats must be integer-valued: rounding is refused."""
    a = np.asarray(x).reshape(-1)
    if a.dtype.kind in "biu":
        return a.astype(np.int64)
    if (a.dtype.kind != "f" or not np.all(np.isfinite(a))
            or not np.all(a == np.round(a))):
        raise ValueError(f"{name} must be integer-valued, got {a.tolist()}")
    return a.astype(np.int64)


def _int_scalar(x, name: str) -> int:
    a = _int_array(x, name)
    if a.size != 1:
        raise ValueError(f"{name} must be a scalar, got {a.size} values")
    return int(a[0])


def topk_2option(scores: np.ndarray, costs_int: np.ndarray,
                 weights_int: np.ndarray, budget_int: int,
                 exact_count: bool = False) -> np.ndarray:
    """O(N log N) exact decode for K=2 with uniform group weights
    (paper Sec 3.1 / App B.1).

    With base cost N*w*c_lo and per-group upgrade step w*(c_hi - c_lo), the
    budget allows n = min(floor((B' - base) / step), N) upgrades, and the
    best choice is the n largest deltas s_hi - s_lo.

    exact_count=False (default, the <= B' problem of Eq.4): at most n
    upgrades, and only those with a strictly positive delta, so the optimum
    matches the full DP. exact_count=True (equality semantics, e.g. expert
    pruning where the budget fixes how many experts are pruned): exactly n
    upgrades, even when some deltas are negative. Ties in the ranking go to
    the lower group index. Non-uniform weights raise; use dp_knapsack.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[1] != 2:
        raise ValueError(f"topk_2option needs scores [N,2], got {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    N = s.shape[0]
    c = _int_array(costs_int, "costs_int")
    w = _int_array(weights_int, "weights_int")
    B = _int_scalar(budget_int, "budget_int")
    if c.shape != (2,) or w.shape != (N,):
        raise ValueError("costs must be [2], weights [N]")
    if B < 0:
        raise ValueError("budget_int must be >= 0")
    if np.any(c < 0) or np.any(w < 0):
        raise ValueError("costs/weights must be nonneg ints")
    if np.any(w != w[0]):
        raise ValueError("topk_2option is exact only for uniform weights; "
                         "use dp_knapsack for non-uniform weights")
    w0 = int(w[0])
    lo_k, hi_k = (0, 1) if c[0] <= c[1] else (1, 0)
    lo_c, hi_c = int(c[lo_k]), int(c[hi_k])
    base = N * w0 * lo_c
    if base > B:
        raise ValueError(f"infeasible: cheapest cost {base} > budget {B}")
    step = w0 * (hi_c - lo_c)
    z = np.zeros((N, 2))
    if step == 0:  # every assignment costs the same: per-row argmax
        if exact_count:
            raise ValueError("exact_count needs a nonzero cost step")
        z[np.arange(N), s.argmax(axis=1)] = 1.0
        return z
    n = min((B - base) // step, N)
    delta = s[:, hi_k] - s[:, lo_k]
    order = np.argsort(-delta, kind="stable")  # largest delta, ties by index
    if not exact_count:
        order = order[delta[order] > 0]
    up = order[:n]
    z[:, lo_k] = 1.0
    z[up, lo_k] = 0.0
    z[up, hi_k] = 1.0
    return z


def _mckp_dp(s: np.ndarray, step: np.ndarray, budget: int) -> np.ndarray:
    """Exact multiple-choice knapsack DP over the budget axis (Eq.4).

    dp[b] is the best value of the groups processed so far with total cost
    <= b, so dp stays nondecreasing in b. Each (group, option) updates all
    budgets at once with numpy slices.
    """
    N, K = s.shape
    B = budget
    dp = np.zeros(B + 1)
    choice = np.zeros((N, B + 1), dtype=np.uint8)
    for i in range(N):
        new = np.full(B + 1, -np.inf)
        for k in range(K):
            cst = int(step[i, k])
            if cst > B:
                continue
            cand = dp[: B + 1 - cst] + s[i, k]
            tgt = new[cst:]
            better = cand > tgt  # strict: the lowest option index keeps ties
            np.copyto(tgt, cand, where=better)
            np.copyto(choice[i, cst:], np.uint8(k), where=better)
        dp = new
    # Smallest budget that reaches the optimum: the cheapest optimal solution.
    b = int(np.argmax(dp >= dp[B]))
    z = np.zeros((N, K))
    for i in range(N - 1, -1, -1):
        k = int(choice[i, b])
        z[i, k] = 1.0
        b -= int(step[i, k])
    return z


def dp_knapsack(scores: np.ndarray, costs_int: np.ndarray,
                weights_int: np.ndarray, budget_int: int,
                use_topk: bool = True) -> np.ndarray:
    """Multiple-choice knapsack, paper Eq.4: one-hot z maximising sum z*s
    with sum_i w_i c_{k(i)} <= B' (integer grid, Sec 3.1), one option per
    group.

    Exact DP vectorised over the budget axis: time O(N*K*B') in numpy,
    choice table N*(B'+1) bytes. Ties are deterministic: the lowest option
    index wins inside the DP, and among equally good assignments the
    cheapest is returned. Costs, weights and budget must be integer-valued;
    nothing is rounded here (use to_int_grid / integer_weights upstream).
    Raises ValueError when the cheapest assignment already exceeds the
    budget. K=2 with uniform weights dispatches to topk_2option, which has
    the same optimal value (ties may resolve differently); use_topk=False
    forces the full DP. CPU-side by design (App B.1).
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] < 1 or s.shape[1] < 1:
        raise ValueError(f"scores must be [N,K], got {s.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    N, K = s.shape
    if K > 255:
        raise ValueError("dp_knapsack supports at most 255 options per group")
    c = _int_array(costs_int, "costs_int")
    w = _int_array(weights_int, "weights_int")
    B = _int_scalar(budget_int, "budget_int")
    if c.shape != (K,) or w.shape != (N,):
        raise ValueError(
            f"costs must be [{K}], weights [{N}], "
            f"got {c.shape}/{w.shape}")
    if B < 0:
        raise ValueError("budget_int must be >= 0")
    if np.any(c < 0) or np.any(w < 0):
        raise ValueError("costs/weights must be nonneg ints")
    cheapest = int((w * c.min()).sum())
    if cheapest > B:
        raise ValueError(f"infeasible: cheapest cost {cheapest} > budget {B}")
    if K == 1:
        return np.ones((N, 1))
    if use_topk and K == 2 and np.all(w == w[0]):
        return topk_2option(s, c, w, B)
    return _mckp_dp(s, w[:, None] * c[None, :], B)
