"""Phase-1 GSQ quantizer stub for Qwen2.5-0.5B (OWNED by GSQ side).

MEMORY RULE: reads ONLY models/Qwen2.5-0.5B/config.json (shapes).
NEVER opens model.safetensors (942MB) — no torch/safetensors/mlx-lm load.
All weights below are synthetic stand-ins; real weight + calibration
wiring is Phase 2 (see block_train.staged_block_plan).

Usage: .venv/bin/python scripts/quantize_qwen.py [--bits 2] [--group-size 64]
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gsq_mlx.gptq import load_qwen_shapes, init_layer, synthetic_weights
from gsq_mlx.quant import get_grid

CONFIG = os.path.join(os.path.dirname(__file__), "..", "models",
                      "Qwen2.5-0.5B", "config.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()

    shapes = load_qwen_shapes(args.config)
    grid = get_grid(args.bits)
    print(f"Qwen2.5-0.5B per-block linears (from config.json only): {shapes}")
    print(f"bits={args.bits} grid={list(map(float, grid))} group_size={args.group_size}")
    total = 0
    for i, (name, (out_f, in_f)) in enumerate(shapes.items()):
        W = synthetic_weights((out_f, in_f), seed=1000 + i)  # stand-in
        init = init_layer(args.bits, (out_f, in_f), args.group_size,
                          seed=i, w=W)
        import numpy as np
        mse = float(((init["w_q"] - W) ** 2).mean())
        sc = init["scales"]
        n = out_f * in_f
        total += n
        print(f"  {name:10s} [{out_f}x{in_f}] params={n:7d} "
              f"groups={sc.size:5d} scale_mean={sc.mean():.5f} rtn_mse={mse:.6f}")
    print(f"per-block params={total} (x24 blocks, embeddings excluded in Phase 2)")
    print("Phase 1 OK: RTN init + GSQ logits built from synthetic weights only.")
    print("Phase 2: load real weights, calibration tokens, staged block loop.")


if __name__ == "__main__":
    main()
