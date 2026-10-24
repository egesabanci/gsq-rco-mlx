"""Export quantized Qwen2.5 linears to an mlx-lm checkpoint (WP4).

  .venv/bin/python scripts/export_mlx.py --out DIR [--shards results/.eval_cache]
  .venv/bin/python scripts/export_mlx.py --out DIR --rtn-bits B      # plain RTN baseline

Shards are the per-layer npz files written by quantize_qwen.py: per linear
<short>.codes (uint8) and <short>.scales (fp16), and per file bits, grid and an
optional group_size (default 64). Linears absent from the shards stay bf16.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from gsq_mlx.e2e import WKEY  # short linear name -> "<block>.<proj>.weight"
from gsq_mlx.export import SafetensorsFile, bits_report, export_mlx_lm
from gsq_mlx.gptq import quantize_rtn
from gsq_mlx.quant import get_grid

DEFAULT_BASE = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DEFAULT_SHARDS = os.path.join(ROOT, "results", ".eval_cache")
GROUP_SIZE = 64


def _module_path(layer: int, short: str) -> str:
    if short not in WKEY:
        raise ValueError(f"unknown linear {short!r} in shards")
    return f"model.layers.{layer}." + WKEY[short][: -len(".weight")]


def load_shards(shard_dir: str) -> dict:
    """layer_XX.npz shards -> {module path: dict(codes, scales, bits, group_size)}."""
    quantized = {}
    for fn in sorted(glob.glob(os.path.join(shard_dir, "layer_*.npz"))):
        m = re.fullmatch(r"layer_(\d+)\.npz", os.path.basename(fn))
        if m is None:
            raise ValueError(f"unexpected shard name {fn}")
        layer = int(m.group(1))
        with np.load(fn) as shard:
            bits = int(shard["bits"])
            if not np.array_equal(shard["grid"], get_grid(bits)):
                raise ValueError(f"{fn}: stored grid differs from get_grid({bits})")
            group_size = int(shard["group_size"]) if "group_size" in shard.files else GROUP_SIZE
            for key in shard.files:
                if key.endswith(".codes"):
                    short = key[: -len(".codes")]
                    quantized[_module_path(layer, short)] = dict(
                        codes=shard[key], scales=shard[f"{short}.scales"],
                        bits=bits, group_size=group_size)
    return quantized


def rtn_native(W, bits: int, group_size: int = GROUP_SIZE, rows: int = 256):
    """gptq.quantize_rtn over blocks of rows (rows are independent), capping its temporaries."""
    codes, scales = [], []
    for r0 in range(0, W.shape[0], rows):
        q = quantize_rtn(W[r0:r0 + rows], bits, group_size)
        codes.append(q["codes"])
        scales.append(q["scales"])
    return np.concatenate(codes), np.concatenate(scales)


def rtn_quantized(base_dir: str, bits: int, group_size: int = GROUP_SIZE) -> dict:
    """Plain RTN for every base linear, as native codes and scales."""
    st = SafetensorsFile(os.path.join(base_dir, "model.safetensors"))
    with open(os.path.join(base_dir, "config.json")) as f:
        n_layers = json.load(f)["num_hidden_layers"]
    quantized = {}
    for layer in range(n_layers):
        for short in WKEY:
            path = _module_path(layer, short)
            codes, scales = rtn_native(st.tensor_f32(path + ".weight"), bits, group_size)
            quantized[path] = dict(codes=codes, scales=scales, bits=bits, group_size=group_size)
    return quantized


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="output directory for the mlx-lm checkpoint")
    ap.add_argument("--base", default=DEFAULT_BASE, help="bf16 base checkpoint directory")
    ap.add_argument("--shards", default=DEFAULT_SHARDS, help="directory of layer_XX.npz shards")
    ap.add_argument("--rtn-bits", type=int, help="export plain RTN at this bit-width instead")
    args = ap.parse_args(argv)

    quantized = (rtn_quantized(args.base, args.rtn_bits) if args.rtn_bits
                 else load_shards(args.shards))
    if not quantized:
        ap.error(f"no quantized linears found in {args.shards}")
    n_params = sum(q["codes"].size for q in quantized.values())
    export_mlx_lm(args.out, quantized, args.base)
    rep = bits_report(quantized, n_params)
    size = os.path.getsize(os.path.join(args.out, "model.safetensors"))
    hist = dict(sorted(Counter(q["bits"] for q in quantized.values()).items()))
    print(f"exported {len(quantized)} linears, {n_params / 1e6:.1f}M weights, "
          f"bits {hist} -> {args.out}")
    print(f"bits/weight: code {rep['code_bpp']:.3f} | native (fp16 scales) "
          f"{rep['native_bpp']:.3f} | MLX export (bf16 scale+bias) {rep['mlx_bpp']:.3f} | "
          f"current shards {rep['shard_bpp']:.3f}")
    print(f"model.safetensors {size / 1e6:.1f} MB (quantized linears alone: "
          f"{rep['mlx_bytes'] / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
