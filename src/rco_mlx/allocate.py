"""RCO bit allocation with the App-B KL objective on the quantized database (paper Eq.6, Alg.1).

rco_mlx.rco.rco_loop owns the Gumbel sampling, the DP solve, the STE, the tangent projection,
the Adam step and the retraction. This module provides what the loop needs from the model:
the grad_fn (KLObjective, one call per Gumbel sample) and the budget on the exact integer grid.

Memory: the assembled model is never materialised. The forward installs one layer's chosen
weights at a time (gsq_rco_common.model.swapped_weights restores the FP arrays), and the
backward runs one decoder block at a time through mx.vjp. Peak memory is the FP model plus
one layer's candidate weights, not the 168 assembled linears.
"""
from __future__ import annotations

from fractions import Fraction

import mlx.core as mx
import numpy as np

from gsq_rco_common.evaluate import CHUNK, kl_value_and_grad_hidden
from gsq_rco_common.model import LINEARS, get_linear_weight, set_linear_weight, swapped_weights
from rco_mlx.manifold import dp_knapsack, expected_cost
from rco_mlx.rco import TAU0, TAU_MIN, integer_weights, rco_loop

MAX_TICKS = 256                    # denominator limit for the exact budget grid
MAX_TABLE_ENTRIES = 100_000_000    # DP choice table: n_groups * (B' + 1) bytes
N_PER_LAYER = len(LINEARS)


def _causal(T: int):
    return "causal" if T > 1 else None


def budget_grid(avg_bits, total_units: int, n_groups: int):
    """Exact budget B = avg_bits * total_units on the integer grid of rco_mlx.rco.to_int_grid.

    total_units is sum(w_i) with w_i = parameter counts / unit (integer_weights). avg_bits is
    read as its shortest decimal (Fraction(repr)), so 2.976 is exactly 2.976 and not a binary
    approximation. ticks is the smallest scale that makes the budget an integer; it is refused
    when the DP table would be too large, rather than rounding the budget.
    Returns (budget, ticks, budget_int).
    """
    frac = Fraction(repr(float(avg_bits))) * int(total_units)
    ticks = frac.denominator
    if ticks > MAX_TICKS:
        raise ValueError(
            f"avg_bits={avg_bits} needs {ticks} ticks per unit on the exact grid "
            f"(max {MAX_TICKS}); use an average with a shorter decimal expansion")
    budget_int = frac.numerator
    if n_groups * (budget_int + 1) > MAX_TABLE_ENTRIES:
        raise ValueError(f"DP table {n_groups} x {budget_int + 1} exceeds {MAX_TABLE_ENTRIES}")
    return float(frac), int(ticks), int(budget_int)


class KLObjective:
    """Mean token KL(p_fp || p_z) over calibration windows (Eq.6) as the RCO grad_fn.

    For a DP assignment z (one-hot, rows i = linears, columns k = options) the linear i is
    assembled as dequant(db[i, k_i]) = W_ref_i + Delta_{i,k_i}. The loss is the mean KL of the
    full model against the FP reference, with the reference hidden states cached once. The
    gradient wrt the hidden state comes from kl_value_and_grad_hidden, and is chained into the
    linear weights block by block with mx.vjp (cotangent = the returned gradient). The STE
    upstream for the loop is dL/dz_ik = <dL/dW_i, Delta_ik> (Frobenius), for every option k,
    evaluated at the assembled weights of this sample.
    """

    def __init__(self, model, bank, windows, *, windows_per_sample: int = 1,
                 chunk: int = CHUNK):
        windows = np.asarray(windows, dtype=np.int32)
        if windows.ndim != 2 or windows.shape[1] < 2:
            raise ValueError(f"windows must be [N, T] with T >= 2, got {windows.shape}")
        if not 1 <= windows_per_sample <= len(windows):
            raise ValueError(f"windows_per_sample must be in 1..{len(windows)}")
        if len(model.model.layers) != bank.n_layers:
            raise ValueError(f"model has {len(model.model.layers)} layers, "
                             f"database {bank.n_layers}")
        self.model = model
        self.bank = bank
        self.windows = windows
        self.m = int(windows_per_sample)
        self.chunk = int(chunk)
        self.n_layers = bank.n_layers
        self.K = len(bank.options)
        self.dtype = model.model.layers[0].self_attn.q_proj.weight.dtype
        self.h_ref = [self.model.model.norm(self._states(windows[w:w + 1], lambda L: None)[-1])
                      for w in range(len(windows))]
        mx.eval(self.h_ref)

    def __call__(self, z, p, step, sample, rng):
        """rco_loop grad_fn: (upstream [n_groups, K], loss) for one Gumbel sample."""
        choice = np.asarray(z).argmax(axis=1)
        ids = sorted(int(w) for w in rng.choice(len(self.windows), size=self.m, replace=False))
        loss, upstream = self.loss_and_upstream(choice, ids)
        return upstream, loss

    def mean_kl(self, choice, window_ids, override=None) -> float:
        """Forward-only loss for checks. override maps group index -> replacement weight."""
        return self.loss_and_upstream(choice, window_ids, override=override,
                                      upstream=False)[0]

    def loss_and_upstream(self, choice, window_ids, override=None, upstream: bool = True):
        choice = np.asarray(choice).astype(int)
        if upstream and override:
            raise ValueError("override is for forward-only checks")
        ids = [int(w) for w in window_ids]
        states = self._states(self.windows[ids],
                              lambda L: self._installed(L, choice, override))
        kl, g_hq = self._kl_grad(states[-1], ids)
        loss = float(kl)
        if not upstream:
            return loss, None
        _, grads = mx.vjp(self.model.model.norm, [states[-1]], [g_hq])
        g = grads[0]
        up = np.zeros((len(self.bank.groups), self.K))
        for L in reversed(range(self.n_layers)):
            cands = [[self.bank.dequant(L * N_PER_LAYER + j, k).astype(self.dtype)
                      for k in range(self.K)] for j in range(N_PER_LAYER)]
            chosen = [cands[j][int(choice[L * N_PER_LAYER + j])] for j in range(N_PER_LAYER)]
            g_in, dW = self._layer_vjp(L, states[L], chosen, g)
            for j, name in enumerate(LINEARS):
                up[L * N_PER_LAYER + j] = self._dots(L, name, dW[j], cands[j])
            g = g_in
            mx.eval(g)
        return loss, up

    # ---------------------------------------------------------------- helpers ---

    def _installed(self, L, choice, override):
        W = {}
        for j, name in enumerate(LINEARS):
            i = L * N_PER_LAYER + j
            w = override[i] if override and i in override else self.bank.dequant(i, int(choice[i]))
            W[name] = w.astype(self.dtype)
        return W

    def _states(self, tokens, layer_weights):
        """Decoder states [embedding, after block 0, ..., after block L-1] (pre final norm)."""
        h = self.model.model.embed_tokens(mx.array(np.asarray(tokens, dtype=np.int32)))
        states = [h]
        mask = _causal(tokens.shape[1])
        for L in range(self.n_layers):
            W = layer_weights(L)
            if W is None:
                h = self.model.model.layers[L](h, mask, None)
            else:
                with swapped_weights(self.model, {(L, n): W[n] for n in LINEARS}):
                    h = self.model.model.layers[L](h, mask, None)
            mx.eval(h)
            states.append(h)
        return states

    def _kl_grad(self, pre_norm, ids):
        hq = self.model.model.norm(pre_norm)
        h_ref = mx.concatenate([self.h_ref[w] for w in ids], axis=0)
        return kl_value_and_grad_hidden(self.model, hq, h_ref, self.chunk)

    def _layer_vjp(self, L, h_in, chosen, cot):
        """Cotangent of block L's input and the gradients of its 7 linear weights."""
        keys = [(L, n) for n in LINEARS]
        layer = self.model.model.layers[L]
        mask = _causal(h_in.shape[1])

        def block(hin, *ws):
            with swapped_weights(self.model, dict(zip(keys, ws))):
                return layer(hin, mask, None)

        _, grads = mx.vjp(block, [h_in, *chosen], [cot])
        return grads[0], grads[1:]

    def _dots(self, L, name, G, cands):
        """<dL/dW, Delta_ik> for every option k; Delta_ik uses the installed (cast) weights."""
        W_ref = get_linear_weight(self.model, L, name).astype(mx.float32)
        G32 = G.astype(mx.float32)
        vals = mx.stack([mx.sum(G32 * (c.astype(mx.float32) - W_ref)) for c in cands])
        return np.asarray(vals, dtype=np.float64)


def initial_logits(bank, init: str) -> np.ndarray:
    """alpha0 = 0 (paper default) or -S / mean(S) from the per-linear output-MSE sensitivities.

    The sensitivity init uses one global scale (the mean over all entries), not one per layer,
    so cross-layer magnitudes are kept.
    """
    n, K = len(bank.groups), len(bank.options)
    if init == "zero":
        return np.zeros((n, K))
    if init == "sensitivity":
        s = np.asarray(bank.sensitivity, dtype=np.float64)
        return -s / float(s.mean())
    raise ValueError(f"init must be 'zero' or 'sensitivity', got {init!r}")


def rco_allocate(objective: KLObjective, bank, avg_bits, *, steps: int, g: int, lr: float,
                 seed: int, tau0: float = TAU0, tau_min: float = TAU_MIN,
                 init: str = "zero", callback=None) -> dict:
    """RCO (Alg.1) with the exact budget. Returns the decoded and continuous results."""
    w_int, _ = integer_weights(bank.counts)
    total = int(w_int.sum())
    bits = np.asarray(bank.options, dtype=np.float64)
    budget, ticks, budget_int = budget_grid(avg_bits, total, len(bank.groups))
    res = rco_loop(initial_logits(bank, init), bits, w_int, budget, steps=steps, g=g,
                   lr=lr, seed=seed, tau0=tau0, tau_min=tau_min, grad_fn=objective,
                   ticks=ticks, callback=callback)
    if res["budget_int"] != budget_int:
        raise RuntimeError(f"grid mismatch: rco_loop {res['budget_int']} vs {budget_int}")
    choice = np.asarray(res["z_final"]).argmax(axis=1)
    decoded_int, decoded_avg = decoded_cost(bank, choice)
    return {
        "choice": choice,
        "history": res["history"],
        "budget": budget,
        "budget_int": budget_int,
        "ticks": ticks,
        "total_units": total,
        "decoded_int_cost": decoded_int,
        "decoded_avg_bits": decoded_avg,
        "expected_avg_bits": float(expected_cost(res["logits"], bits, w_int)) / total,
    }


def decoded_cost(bank, choice) -> tuple[int, float]:
    """Integer cost sum_i w_i b_{k_i} and decoded average bits = cost / total_int_weight."""
    w_int, _ = integer_weights(bank.counts)
    cost = int(np.sum(w_int * np.asarray(bank.options, dtype=np.int64)[np.asarray(choice)]))
    return cost, cost / int(w_int.sum())


def proxy_dp_choice(bank, avg_bits) -> np.ndarray:
    """Baseline: DP on the per-linear output-MSE sensitivities of the same database."""
    w_int, _ = integer_weights(bank.counts)
    budget, ticks, budget_int = budget_grid(avg_bits, int(w_int.sum()), len(bank.groups))
    costs_int = np.asarray(bank.options, dtype=np.int64) * ticks
    z = dp_knapsack(-np.asarray(bank.sensitivity, dtype=np.float64), costs_int, w_int, budget_int)
    return np.asarray(z).argmax(axis=1)


def uniform_choice(bank, bits: int) -> np.ndarray:
    """Baseline: every linear at the same bit-width (must be one of the options)."""
    if int(bits) not in bank.options:
        raise ValueError(f"{bits} bits is not among the database options {bank.options}")
    return np.full(len(bank.groups), bank.options.index(int(bits)), dtype=np.int64)


def install_choice(model, bank, choice) -> None:
    """Destructive: replace every linear by its assembled database weight.

    For evaluation only (the FP arrays are released). The objective never calls this.
    """
    for i, (L, name) in enumerate(bank.groups):
        set_linear_weight(model, L, name, bank.dequant(i, int(choice[i])))


def allocation_summary(bank, choice):
    """Per-linear bits and per-layer parameter-weighted average bits of an assignment."""
    bits = np.asarray(bank.options)[np.asarray(choice)]
    per_linear = {f"{L}.{name}": int(b) for (L, name), b in zip(bank.groups, bits)}
    per_layer = {}
    for L in range(bank.n_layers):
        sl = slice(L * N_PER_LAYER, (L + 1) * N_PER_LAYER)
        n = bank.counts[sl]
        per_layer[str(L)] = {"avg_bits": float((n * bits[sl]).sum() / n.sum()),
                             "params": int(n.sum())}
    return per_linear, per_layer


def truncate_layers(model, n_layers: int):
    """Keep the first n_layers decoder blocks (the rest are released)."""
    layers = model.model.layers
    if not 1 <= n_layers <= len(layers):
        raise ValueError(f"n_layers must be in 1..{len(layers)}, got {n_layers}")
    if n_layers < len(layers):
        model.model.layers = layers[:n_layers]
    return model
