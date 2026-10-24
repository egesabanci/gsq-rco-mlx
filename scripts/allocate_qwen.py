"""Phase-1 RCO allocator stub for Qwen2.5-0.5B (OWNED by RCO side).

MEMORY RULE: reads ONLY models/Qwen2.5-0.5B/config.json (shapes).
NEVER opens model.safetensors (942MB) — no torch/safetensors/mlx-lm load.
All sensitivities below are synthetic stand-ins; the real App-B KL
objective (calibration tokens via mlx_lm) + GPTQ-residual weight assembly
is Phase-2 work (see rco_mlx/rco.py).

Target (Phase 2): per-layer bits over Qwen2.5-0.5B with options [2,3,4,8].

Usage: .venv/bin/python scripts/allocate_qwen.py [--avg-bits 3.0] [--steps 10]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

from rco_mlx.manifold import assignment_cost
from rco_mlx.rco import rco_loop, tau_schedule

CONFIG = os.path.join(os.path.dirname(__file__), "..", "models",
                      "Qwen2.5-0.5B", "config.json")
OPTIONS = [2, 3, 4, 8]
GROUPS_PER_LAYER = 4  # attn, gate, up, down (abstract Phase-1 grouping)


def load_qwen_dims(config_path: str = CONFIG) -> dict:
    """Read ONLY config.json (never the safetensors)."""
    with open(config_path) as f:
        c = json.load(f)
    head_dim = c["hidden_size"] // c["num_attention_heads"]
    return {
        "layers": c["num_hidden_layers"],
        "hidden": c["hidden_size"],
        "intermediate": c["intermediate_size"],
        "heads": c["num_attention_heads"],
        "kv_heads": c["num_key_value_heads"],
        "head_dim": head_dim,
        "kv_dim": c["num_key_value_heads"] * head_dim,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--avg-bits", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()

    dims = load_qwen_dims(args.config)
    L = dims["layers"]
    N = L * GROUPS_PER_LAYER
    costs = np.array(OPTIONS, dtype=np.float64)
    weights = np.ones(N)
    budget = args.avg_bits * N
    print(f"Qwen2.5-0.5B dims (from config.json only): {dims}")
    print(f"groups: {L} layers x {GROUPS_PER_LAYER} = {N}, "
          f"options={OPTIONS}, budget={budget:.1f} (avg {args.avg_bits})")

    # Synthetic sensitivity init: deeper layers slightly prefer higher bits
    # (stand-in; Phase 2 inits from heuristic scores, then retracts).
    rng = np.random.default_rng(args.seed)
    logits = rng.standard_normal((N, len(OPTIONS))) * 0.5
    layer_bias = (np.arange(N) // GROUPS_PER_LAYER) / max(1, L - 1) - 0.5
    logits[:, -1] += 0.3 * layer_bias
    logits[:, 0] -= 0.3 * layer_bias

    res = rco_loop(logits, costs, weights, budget, steps=args.steps,
                   g=args.samples, lr=args.lr, seed=args.seed)
    for h in res["history"]:
        print(f"  step {h['t']:2d} tau={h['tau']:.4f} cost={h['cost']:.6f} "
              f"(target {budget:.1f})")
    z = res["z_final"]
    per_layer = (z.reshape(L, GROUPS_PER_LAYER, -1)
                 * np.array(OPTIONS)[None, None, :]).sum(axis=(1, 2)) / GROUPS_PER_LAYER
    for li, b in enumerate(per_layer):
        print(f"  layer {li:2d} avg bits: {b:.2f}")
    print(f"decoded total={res['decoded_cost']} B'={res['budget_int']} "
          f"overall avg={res['decoded_cost'] / N:.3f}")
    print("Phase 1 OK: synthetic RCO allocation only.")
    print("Phase 2: real KL via mlx_lm calibration + GPTQ-residual assembly.")


if __name__ == "__main__":
    main()
