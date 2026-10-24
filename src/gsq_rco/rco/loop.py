"""RCO loop skeleton for MLX (paper Alg.1, Sec 3 + App B/F).

Phase 1: synthetic-only. No model weights, no mlx_lm, no safetensors —
only tiny synthetic logits / KL stubs so Mac memory stays safe.
Real KL (calibration tokens via mlx_lm) + weight assembly via GPTQ
residuals is Phase-2 work (see assemble_weights_stub docstring).

One step (paper Alg.1 lines 2-15):
  forward  (lines 4-8):  g independent Gumbel samples; perturbed logits
           h^(j) = (a + G^(j))/tau; z^(j) = DP-solve(h^(j)) (Eq.4);
           STE surrogate sg(z^(j) - p^(j)) + p^(j) with p^(j) the softmax
           of the SAME perturbed logits (bias -> 0 as tau -> 0).
  project  (lines 9-10): g <- g - (<g,n>/||n||^2) n  (removes the
           constraint-normal STE bias, Prop.6).
  step     (lines 11-12): Adam-style step, then binary-search retract
           C(a + t*c) = B (Prop.3).
  transport(lines 13-14): m <- m - (<m,n'>/||n'||^2) n' (first moment only).

Hyperparameters (App F sweeps): exp tau anneal 1.0 -> 0.01
  tau_t = max(tau_min, tau0 * (tau_min/tau0)^{t/(T-1)}),
g = 4 samples (sample efficiency dominates step count), Adam-style
step lr 0.05-0.2 (default 0.1).

Dual backend like manifold.py: numpy default; pass use_mx=True (needs
mlx) to run the continuous path on mx arrays. DP decode stays CPU-side
(App B.1) in both modes.
"""
from __future__ import annotations
import numpy as np

from gsq_rco.rco.manifold import (
    softmax, expected_cost, normal_vector, project_tangent, transport,
    retract, dp_knapsack, assignment_cost, assignment_value,
)

try:
    import mlx.core as mx  # type: ignore
    _HAS_MLX = True
except Exception:
    mx = None  # type: ignore
    _HAS_MLX = False

# Paper App F defaults.
TAU0, TAU_MIN = 1.0, 0.01
G_DEFAULT = 4
LR_DEFAULT = 0.1
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8


def _is_mx(x) -> bool:
    return _HAS_MLX and isinstance(x, mx.array)


# -------------------------------------------------------------- schedule ---

def tau_schedule(t: int, T: int, tau0: float = TAU0,
                 tau_min: float = TAU_MIN) -> float:
    """Exponential anneal tau0 -> tau_min over T steps (paper Sec 3.1).

    tau_t = max(tau_min, tau0 * (tau_min/tau0)^{t/(T-1)}), so step 0 is
    tau0 (diverse DP samples, diffuse surrogate) and the last step is
    tau_min (STE bias tight). Endpoint convention t/(T-1) matches the
    test_smoke's 1.0*(0.01)**(t/9) for T=10.
    """
    if T <= 1:
        return float(tau_min)
    frac = min(max(t, 0), T - 1) / (T - 1)
    return max(float(tau_min), float(tau0) * (float(tau_min) / float(tau0)) ** frac)


# --------------------------------------------------------------- sampling ---

def sample_gumbel(shape, rng=None, key=None):
    """Standard Gumbel(0,1) noise. Numpy if rng given (or no mlx),
    else mx.random.gumbel (optionally with explicit key)."""
    if key is not None and _HAS_MLX:
        return mx.random.gumbel(shape=list(shape), key=key)
    if rng is not None or not _HAS_MLX:
        r = rng or np.random.default_rng(0)
        u = r.random(tuple(shape)).astype(np.float64)
        return (-np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))).astype(np.float64)
    return mx.random.gumbel(shape=list(shape))


def perturbed_logits(logits, gumbel, tau: float):
    """Pure core h = (a + G)/tau (Alg.1 line 5). No RNG inside, so it is
    directly usable under mx.grad when inputs are mx arrays."""
    tau = max(float(tau), 1e-6)
    return (logits + gumbel) / tau


def soft_probs(h):
    """Softmax of the SAME perturbed logits that produced z (Alg.1 line 7).
    Decoupling the surrogate (unperturbed softmax) would suppress the
    sampled-mode concentration; bias vanishes as tau -> 0."""
    return softmax(h)


def ste_grad(upstream, p, tau: float):
    """STE backward: upstream dL/dy (evaluated at forward value y = z)
    through the softmax Jacobian at the perturbed logits, with the 1/tau
    from h = (a+G)/tau (Alg.1 line 7: contributes 1/tau).

    J^T v = p * (v - <p,v>) — no materialized Jacobian. Pure + grad-safe.
    """
    tau = max(float(tau), 1e-6)
    if _is_mx(upstream) or _is_mx(p):
        inner = mx.sum(p * upstream, axis=1, keepdims=True)
        return (p * (upstream - inner)) / tau
    up = np.asarray(upstream, dtype=np.float64)
    pp = np.asarray(p, dtype=np.float64)
    inner = (pp * up).sum(axis=1, keepdims=True)
    return (pp * (up - inner)) / tau


# ------------------------------------------------------------ int grid ---

def to_int_grid(costs, weights, budget, ticks: int = 1,
                allow_rounding: bool = False):
    """Map float costs/budget to the DP integer grid (paper Sec 3.1).

    costs_int = round(costs * ticks), budget_int = round(budget * ticks).
    Weights must already be integer-valued (build them with
    integer_weights): a fractional weight raises unless allow_rounding=True,
    because rounding it silently changes the budget being enforced. For bit
    options [2,3,4,8] with ticks=1 the grid is exact.
    """
    c = np.asarray(costs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    wr = np.round(w)
    if not allow_rounding and not np.array_equal(wr, w):
        raise ValueError(
            "to_int_grid: weights are not integer-valued; pass "
            "integer_weights(counts) or allow_rounding=True (rounding changes "
            "the budget being enforced)")
    ci = np.round(c * int(ticks)).astype(np.int64)
    wi = wr.astype(np.int64)
    Bi = int(round(float(budget) * int(ticks)))
    return ci, wi, Bi


def integer_weights(counts):
    """Exact integer group weights: parameter counts divided by their gcd.

    Returns (w_int, unit) with counts == w_int * unit, no rounding. For the
    Qwen2.5-0.5B linear layers (q/o 802816, k/v 114688, gate/up/down
    4358144) unit = 114688 and w_int = [7, 1, 1, 7, 38, 38, 38].
    """
    raw = np.asarray(counts)
    if raw.size == 0:
        raise ValueError("counts must be non-empty")
    if raw.dtype.kind == "f":
        if not (np.all(np.isfinite(raw)) and np.all(raw == np.round(raw))):
            raise ValueError("counts must be integer-valued")
    elif raw.dtype.kind not in "iu":
        raise ValueError(f"counts must be numeric, got dtype {raw.dtype}")
    ints = raw.astype(np.int64)
    if np.any(ints < 0):
        raise ValueError("counts must be nonnegative")
    unit = int(np.gcd.reduce(ints.reshape(-1)))
    if unit == 0:
        raise ValueError("all counts are zero")
    return ints // unit, unit


# --------------------------------------------- synthetic KL + assembly ---


# ------------------------------------------------------------------ loop ---

def _unpack_grad(out):
    """grad_fn returns dL/dy, or (dL/dy, loss) to log a per-sample loss."""
    if isinstance(out, tuple):
        up, loss = out
        return up, float(loss)
    return out, None


def rco_loop(logits_init, costs, weights, budget, steps: int = 10,
             g: int = G_DEFAULT, lr: float = LR_DEFAULT, seed: int = 0,
             tau0: float = TAU0, tau_min: float = TAU_MIN,
             beta1: float = ADAM_BETAS[0], beta2: float = ADAM_BETAS[1],
             eps: float = ADAM_EPS, grad_fn=None, use_mx: bool = False,
             ticks: int = 1, callback=None):
    """RCO (paper Alg.1). Returns dict with logits, momentum, velocity,
    per-step history [{t, tau, cost[, loss]}], final DP assignment + costs.

    grad_fn(z, p, step, sample, rng) is required (ValueError if None: a
    random default would silently optimise noise). It returns the upstream
    dL/dy at the forward value y = z (STE: forward z, backward p), or a
    tuple (dL/dy, loss); with a loss, history[t]["loss"] is the mean over
    the g samples. Adam-style step (default lr 0.1, App F range 0.05-0.2)
    with bias correction; retraction after every step keeps |C-B| within
    tolerance; momentum transport keeps m tangent. callback(t, logits,
    record), if given, runs after each step.
    """
    if grad_fn is None:
        raise ValueError(
            "rco_loop needs grad_fn(z, p, step, sample, rng) returning dL/dy "
            "(or (dL/dy, loss)); there is no silent random default")
    c_np = np.asarray(costs, dtype=np.float64)
    w_np = np.asarray(weights, dtype=np.float64)
    B = float(budget)
    ci, wi, Bi = to_int_grid(c_np, w_np, B, ticks)
    rng = np.random.default_rng(seed)

    if use_mx and not _HAS_MLX:
        raise RuntimeError("use_mx=True but mlx is not installed")

    if use_mx:
        a = mx.array(np.asarray(logits_init, dtype=np.float32))
        m = mx.zeros_like(a)
        v = mx.zeros_like(a)
        a = retract(a, c_np, w_np, B)
        mx.eval(a)
    else:
        a = np.asarray(logits_init, dtype=np.float64).copy()
        a = retract(a, c_np, w_np, B)
        m = np.zeros_like(a)
        v = np.zeros_like(a)

    history = []
    for t in range(steps):
        tau = tau_schedule(t, steps, tau0, tau_min)
        losses = []
        if use_mx:
            g_acc = mx.zeros_like(a)
            for j in range(g):
                Gj = mx.random.gumbel(shape=list(a.shape))
                h = perturbed_logits(a, Gj, tau)
                mx.eval(h)
                z = dp_knapsack(np.asarray(h, dtype=np.float64), ci, wi, Bi)
                p = soft_probs(h)
                mx.eval(p)
                up, loss = _unpack_grad(
                    grad_fn(z, np.asarray(p, dtype=np.float64), t, j, rng))
                if loss is not None:
                    losses.append(loss)
                up_np = np.asarray(up, dtype=np.float64)
                up = mx.array(up_np.astype(np.float32))
                g_acc = g_acc + ste_grad(up, p, tau)
            g_avg = g_acc / g
            n = normal_vector(a, c_np, w_np)
            gp = project_tangent(g_avg, n)
            m = beta1 * m + (1 - beta1) * gp
            v = beta2 * v + (1 - beta2) * gp * gp
            mhat = m / (1 - beta1 ** (t + 1))
            vhat = v / (1 - beta2 ** (t + 1))
            a = a - lr * mhat / (mx.sqrt(vhat) + eps)
            a = retract(a, c_np, w_np, B)
            n2 = normal_vector(a, c_np, w_np)
            m = transport(m, n2)
            mx.eval(a, m, v)
            cv = expected_cost(a, c_np, w_np)
            mx.eval(cv)
            cost = float(cv.item())
        else:
            g_acc = np.zeros_like(a)
            for j in range(g):
                Gj = sample_gumbel(a.shape, rng=rng)
                h = perturbed_logits(a, Gj, tau)
                z = dp_knapsack(h, ci, wi, Bi)
                p = np.asarray(soft_probs(h))
                up, loss = _unpack_grad(grad_fn(z, p, t, j, rng))
                if loss is not None:
                    losses.append(loss)
                g_acc = g_acc + ste_grad(up, p, tau)
            g_avg = g_acc / g
            n = normal_vector(a, c_np, w_np)
            gp = project_tangent(g_avg, n)
            m = beta1 * m + (1 - beta1) * gp
            v = beta2 * v + (1 - beta2) * gp * gp
            mhat = m / (1 - beta1 ** (t + 1))
            vhat = v / (1 - beta2 ** (t + 1))
            a = a - lr * mhat / (np.sqrt(vhat) + eps)
            a = retract(a, c_np, w_np, B)
            m = transport(m, normal_vector(a, c_np, w_np))
            cost = expected_cost(a, c_np, w_np)
        if losses and len(losses) != g:
            raise ValueError("grad_fn returned a loss for some samples only")
        record = {"t": t, "tau": tau, "cost": cost}
        if losses:
            record["loss"] = float(np.mean(losses))
        history.append(record)
        if callback is not None:
            callback(t, a, record)

    if use_mx:
        mx.eval(a)
        a_np = np.asarray(a, dtype=np.float64)
    else:
        a_np = a
    z_final = dp_knapsack(a_np, ci, wi, Bi)
    return {
        "logits": a,
        "momentum": m,
        "velocity": v,
        "history": history,
        "z_final": z_final,
        "decoded_cost": assignment_cost(z_final, ci, wi),
        "decoded_value": assignment_value(z_final, a_np),
        "budget": B,
        "budget_int": Bi,
        "costs_int": ci,
        "weights_int": wi,
    }
