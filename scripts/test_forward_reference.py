"""Forward-reference check for the block forwards (guards audit finding #2).

For layers 0 and 12 on a real WikiText-2 test window (T = 512 tokens), each block
implementation must
  1. match mlx-lm's TransformerBlock on the block UPDATE (out - in): relative error
     below 3%, and
  2. be causal: changing the input at position t must leave the outputs at
     positions < t unchanged (probed at several t).

Implementations, reported separately:
  e2e      gsq_mlx.e2e.Qwen2Stream.block, weights read by its LazyWeights reader.
           Must pass.
  phase2a  rco_mlx.phase2a.layer_forward, if it still exists. Expected to FAIL
           today: its GQA expansion interleaves positions with heads.

Exit code 0 only if every present implementation passes all of its checks.
Usage: .venv/bin/python scripts/test_forward_reference.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask

from gsq_mlx.e2e import BKEY, LIN_ORDER, WKEY, LazyWeights, Qwen2Stream
from gsq_rco_common.data import MODEL_DIR, eval_windows
from gsq_rco_common.model import load_fp_model

LAYERS = (0, 12)
SEQ = 512
REL_TOL = 0.03
CAUSAL_TOL = 1e-3
PROBES = (1, 7, 128, 300, SEQ - 1)


def layer_inputs(model, tokens) -> dict:
    """Residual-stream input of each probed layer, from mlx-lm's own forward."""
    h = model.model.embed_tokens(mx.array(tokens[None]))
    mask = create_attention_mask(h, None)
    inputs = {}
    for i in range(max(LAYERS) + 1):
        if i in LAYERS:
            inputs[i] = h[0]
        h = model.model.layers[i](h, mask)
    mx.eval(list(inputs.values()))
    return inputs


def ref_block(model, i: int, x: mx.array) -> mx.array:
    """mlx-lm TransformerBlock i on one sequence [T, H], with its causal mask."""
    xb = x[None]
    return model.model.layers[i](xb, create_attention_mask(xb, None))[0]


def e2e_block_weights(lw: LazyWeights, i: int) -> dict:
    """Block weights in the layout Qwen2Stream expects (as quantize_qwen loads them)."""
    P = f"model.layers.{i}."
    W = {short: lw.load(P + WKEY[short]) for short in LIN_ORDER}
    W.update({short: lw.load(P + full) for short, full in BKEY.items()})
    W["in_norm"] = lw.load(P + "input_layernorm.weight")
    W["post_norm"] = lw.load(P + "post_attention_layernorm.weight")
    mx.eval(*W.values())
    return W


def rel_update_error(out, ref_out, x) -> float:
    """||(out - x) - (ref_out - x)|| / ||ref_out - x||, in float64."""
    def f(a):
        return np.asarray(a.astype(mx.float32), dtype=np.float64)
    d_ref = f(ref_out) - f(x)
    return float(np.linalg.norm(f(out) - f(x) - d_ref) / np.linalg.norm(d_ref))


def causal_probe(run, i: int, x: mx.array, t: int):
    """Perturb input position t only. Returns (largest relative change
    ||out_p' - out_p|| / ||out_p|| over positions p < t, relative change at
    position t). The per-position form keeps the massive first-token activations
    from hiding leakage; the second number shows the probe actually moved the
    output, so a zero leak is not vacuous."""
    xn = np.asarray(x.astype(mx.float32), dtype=np.float32).copy()
    base = np.asarray(run(i, x).astype(mx.float32))
    rng = np.random.default_rng(1000 + t)
    xn[t] += xn.std() * rng.standard_normal(xn.shape[1]).astype(np.float32)
    pert = np.asarray(run(i, mx.array(xn).astype(mx.bfloat16)).astype(mx.float32))
    num = np.linalg.norm(pert[:t] - base[:t], axis=-1)
    den = np.linalg.norm(base[:t], axis=-1)
    own = np.linalg.norm(pert[t] - base[t]) / np.linalg.norm(base[t])
    return float((num / den).max()), float(own)


def evaluate_impl(name: str, run, inputs: dict, ref_outs: dict) -> bool:
    print(f"== {name} ==", flush=True)
    ok = True
    for i in LAYERS:
        x = inputs[i]
        rel = rel_update_error(run(i, x), ref_outs[i], x)
        probes = [causal_probe(run, i, x, t) for t in PROBES]
        worst = max(leak for leak, _ in probes)
        moved = min(own for _, own in probes)
        p_rel = rel < REL_TOL
        p_causal = worst < CAUSAL_TOL and moved > CAUSAL_TOL
        ok = ok and p_rel and p_causal
        print(f"  layer {i:2d}  block-update rel.err {rel:.4f} (tol {REL_TOL})"
              f"  {'PASS' if p_rel else 'FAIL'}")
        print(f"  layer {i:2d}  causality max leak {worst:.2e} over t={PROBES}"
              f" (tol {CAUSAL_TOL:g}); probe moved t by >= {moved:.2e}"
              f"  {'PASS' if p_causal else 'FAIL'}")
    print(f"  RESULT {name}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main() -> int:
    tokens = eval_windows(SEQ, "test", max_windows=1)[0]
    model = load_fp_model()
    args = model.args
    inputs = layer_inputs(model, tokens)
    ref_outs = {i: ref_block(model, i, inputs[i]) for i in LAYERS}
    mx.eval(list(ref_outs.values()))

    results = {}
    lw = LazyWeights(str(MODEL_DIR / "model.safetensors"))
    e2e_W = {i: e2e_block_weights(lw, i) for i in LAYERS}
    lw.close()
    qs = Qwen2Stream(n_heads=args.num_attention_heads,
                     n_kv=args.num_key_value_heads,
                     theta=args.rope_theta, eps=args.rms_norm_eps)
    results["e2e"] = evaluate_impl(
        "e2e: gsq_mlx.e2e.Qwen2Stream.block",
        lambda i, x: qs.block(x, e2e_W[i]), inputs, ref_outs)

    try:
        from rco_mlx import phase2a
        layer_forward = phase2a.layer_forward
    except (ImportError, AttributeError) as e:
        print(f"== phase2a: SKIPPED, rco_mlx.phase2a.layer_forward not "
              f"available ({type(e).__name__}: {e}) ==", flush=True)
    else:
        cfg = phase2a.load_cfg(str(MODEL_DIR / "config.json"))
        full = dict(tree_flatten(model.parameters()))

        def run_phase2a(i, x):
            T = x.shape[0]
            cos, sin = phase2a.rope_cache(T, cfg["head_dim"], cfg["rope_theta"])
            return layer_forward(x, phase2a.split_layer(full, i), cfg,
                                 cos, sin, phase2a.causal_mask(T))

        results["phase2a"] = evaluate_impl(
            "phase2a: rco_mlx.phase2a.layer_forward", run_phase2a, inputs, ref_outs)

    summary = ", ".join(f"{k}={'PASS' if ok else 'FAIL'}" for k, ok in results.items())
    print(f"SUMMARY: {summary}")
    if results.get("phase2a") is False:
        print("phase2a FAIL is the known finding #2 (GQA head/position interleaving).")
    return 1 if not all(results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
