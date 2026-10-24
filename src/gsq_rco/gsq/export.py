"""MLX export of GSQ/RCO quantized linears to an mlx-lm checkpoint (WP4).

Native format, per linear W[out, in] (gsq.pipeline shards, RTN):
    codes   uint8 [out, in]     index into quant.get_grid(bits)
    scales  float [out, in/g]   per-group scale, may be negative
    W[o, i] = scales[o, i // g] * grid[codes[o, i]]

MLX affine format (mx.quantize / mx.dequantize, mode="affine"):
    W[o, i] = scale[o, i // g] * q[o, i] + bias[o, i // g],   q in [0, 2^b),
with q packed LSB-first into little-endian uint32 words along the input dim.
For b >= 2, grid[c] = c - 2^(b-1), so q = c, scale = s and bias = -2^(b-1) * s
give s * grid[c] exactly. Ternary (bits=1, grid [-1, 0, 1]) is stored as 2-bit
with q = c + 1, i.e. the same values inside the 2-bit grid {-2, -1, 0, 1}.

Scales and biases are stored in bf16 by default: mlx-lm keeps the stored dtype
on load and mx.quantize uses the input dtype. bias = -2^(b-1) * scale is a
power-of-two multiple, so bf16(bias) = -2^(b-1) * bf16(scale) and the loaded
weight is bf16(s) * grid[c]; the only error is bf16 rounding of s, relative
2^-8 (bf16 unit roundoff: 8 significant bits).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import shutil
import struct
from collections import Counter

import mlx.core as mx
import numpy as np

from gsq_rco.gsq.e2e import WKEY
from gsq_rco.gsq.gptq import quantize_rtn
from gsq_rco.gsq.quant import get_grid

MLX_BITS = (2, 3, 4, 5, 6, 8)
GROUP_SIZES = (32, 64, 128)
SCALE_DTYPES = {  # name -> (mx dtype, safetensors dtype, bits per element)
    "bfloat16": (mx.bfloat16, "BF16", 16),
    "float16": (mx.float16, "F16", 16),
    "float32": (mx.float32, "F32", 32),
}
_ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1}
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "generation_config.json")


def mlx_bits(bits: int) -> int:
    """MLX storage width for a native bit-width (ternary is stored as 2-bit)."""
    if bits == 1:
        return 2
    if bits in MLX_BITS:
        return bits
    raise ValueError(f"no MLX affine format for {bits} bits")


def pack_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    """uint8 codes [out, in] (< 2^bits) -> uint32 [out, in*bits/32], MLX layout.

    Value j of a row occupies stream bits [j*bits, (j+1)*bits), LSB first; the
    stream is read as little-endian 32-bit words (checked against mx.quantize).
    """
    out_f, in_f = codes.shape
    if (in_f * bits) % 32:
        raise ValueError(f"{in_f} inputs x {bits} bits is not a whole number of words")
    planes = (codes[..., None] >> np.arange(bits, dtype=np.uint8)) & 1
    stream = np.packbits(planes.reshape(out_f, in_f * bits), axis=1, bitorder="little")
    return np.ascontiguousarray(stream).view("<u4")


def to_mlx_affine(codes, scales, bits, group_size=64, scale_dtype="bfloat16"):
    """Native (codes, scales) -> (packed uint32 weight, scales, biases) in MLX affine form."""
    out_f, in_f = codes.shape
    if codes.dtype != np.uint8 or codes.max(initial=0) >= len(get_grid(bits)):
        raise ValueError(f"codes must be uint8 indices below {len(get_grid(bits))} levels")
    if group_size not in GROUP_SIZES or in_f % group_size:
        raise ValueError(f"group_size {group_size} must be in {GROUP_SIZES} and divide {in_f}")
    if np.shape(scales) != (out_f, in_f // group_size):
        raise ValueError(f"scales shape {np.shape(scales)} != {(out_f, in_f // group_size)}")
    mb = mlx_bits(bits)
    q = codes + np.uint8(1) if bits == 1 else codes
    s32 = np.asarray(scales, dtype=np.float32)
    dtype = SCALE_DTYPES[scale_dtype][0]
    packed = pack_codes(q, mb)
    sc = mx.array(s32).astype(dtype)
    bi = mx.array(np.float32(-(1 << (mb - 1))) * s32).astype(dtype)
    return packed, sc, bi


class SafetensorsFile:
    """Header parser with per-tensor byte reads, so no full-file load is needed."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        self.header = header
        self._data_start = 8 + n

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def names(self) -> list[str]:
        """Tensor names in file order."""
        return sorted(self.header, key=lambda k: self.header[k]["data_offsets"][0])

    def dtype(self, name: str) -> str:
        return self.header[name]["dtype"]

    def shape(self, name: str) -> list[int]:
        return list(self.header[name]["shape"])

    def read_bytes(self, name: str) -> bytes:
        begin, end = self.header[name]["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self._data_start + begin)
            return f.read(end - begin)

    def tensor_f32(self, name: str) -> np.ndarray:
        """A BF16 tensor widened exactly to float32."""
        if self.dtype(name) != "BF16":
            raise TypeError(f"{name} is {self.dtype(name)}, expected BF16")
        u = np.frombuffer(self.read_bytes(name), dtype="<u2").astype(np.uint32) << 16
        return u.view(np.float32).reshape(self.shape(name))


def _mx_bytes(a: mx.array) -> bytes:
    if a.dtype == mx.bfloat16:  # bit view keeps the bf16 pattern
        return np.array(a.view(mx.uint16)).astype("<u2").tobytes()
    return np.array(a).astype("<f2" if a.dtype == mx.float16 else "<f4").tobytes()


def export_mlx_lm(out_dir, quantized, base_model_dir, scale_dtype="bfloat16"):
    """Write an mlx-lm checkpoint: model.safetensors, config.json, tokenizer files.

    quantized: {module path: dict(codes, scales, bits, group_size)} with the path
    being the weight name without ".weight" (e.g. "model.layers.3.mlp.down_proj").
    Every other base tensor is copied byte for byte. Tensors are streamed one at a
    time, so peak memory is one linear plus the codes passed in.
    """
    if not quantized:
        raise ValueError("no quantized linears to export")
    base = SafetensorsFile(os.path.join(base_model_dir, "model.safetensors"))
    for path, q in quantized.items():
        if path + ".weight" not in base or base.shape(path + ".weight") != list(q["codes"].shape):
            raise ValueError(f"{path}: not a base linear of shape {q['codes'].shape}")
    st_dtype = SCALE_DTYPES[scale_dtype][1]

    plan = []  # (name, safetensors dtype, shape) in file order
    for name in base.names():
        path = name[: -len(".weight")] if name.endswith(".weight") else None
        if path is not None and path in quantized:
            q = quantized[path]
            out_f, in_f = q["codes"].shape
            gshape = [out_f, in_f // q["group_size"]]
            plan += [(name, "U32", [out_f, in_f * mlx_bits(q["bits"]) // 32]),
                     (path + ".scales", st_dtype, gshape),
                     (path + ".biases", st_dtype, gshape)]
        else:
            plan.append((name, base.dtype(name), base.shape(name)))

    header = {"__metadata__": {"format": "mlx"}}
    sizes, off = [], 0
    for name, dtype, shape in plan:
        n = math.prod(shape) * _ITEMSIZE[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + n]}
        sizes.append(n)
        off += n
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)

    def chunks():
        for name in base.names():
            path = name[: -len(".weight")] if name.endswith(".weight") else None
            if path is None or path not in quantized:
                yield base.read_bytes(name)
                continue
            q = quantized[path]
            packed, sc, bi = to_mlx_affine(q["codes"], q["scales"], q["bits"],
                                           q["group_size"], scale_dtype)
            yield packed.tobytes()
            yield _mx_bytes(sc)
            yield _mx_bytes(bi)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model.safetensors")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for (name, _, _), n, data in zip(plan, sizes, chunks()):
            if len(data) != n:
                raise RuntimeError(f"{name}: wrote {len(data)} bytes, header says {n}")
            f.write(data)

    with open(os.path.join(base_model_dir, "config.json")) as f:
        config = json.load(f)
    per_module = {p: {"group_size": q["group_size"], "bits": mlx_bits(q["bits"])}
                  for p, q in quantized.items()}
    default = {
        "group_size": Counter(v["group_size"] for v in per_module.values()).most_common(1)[0][0],
        "bits": Counter(v["bits"] for v in per_module.values()).most_common(1)[0][0],
    }
    config["quantization"] = {**default, **per_module}
    config["quantization_config"] = config["quantization"]
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for fn in TOKENIZER_FILES:
        src = os.path.join(base_model_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, out_dir)
    return out_path


def bits_report(quantized, n_quantized_params, native_scale_bits=16, export_scale_bits=16):
    """Bits per quantized weight under four accountings (scale overhead included).

    code_bpp    log2(#grid levels): ternary 1.585 (paper: "1.58-bit"), 2-bit 2, ...
    native_bpp  code_bpp + native_scale_bits per group (paper Tab. 7 counts this)
    mlx_bpp     MLX bits (ternary stored as 2) + scale and bias bits per group
    shard_bpp   current shard storage: uint8 codes + fp16 scales
    Raises if n_quantized_params is not the sum of the codes' sizes.
    """
    n = groups = 0
    code_bits = stored_bits = 0.0
    for q in quantized.values():
        cnt = q["codes"].size
        n += cnt
        groups += q["scales"].size
        code_bits += cnt * math.log2(len(get_grid(q["bits"])))
        stored_bits += cnt * mlx_bits(q["bits"])
    if n != n_quantized_params:
        raise ValueError(f"codes hold {n} weights, caller says {n_quantized_params}")
    native_total = code_bits + native_scale_bits * groups
    mlx_total = stored_bits + 2 * export_scale_bits * groups
    return {
        "n_params": n,
        "code_bpp": code_bits / n,
        "native_bpp": native_total / n,
        "mlx_bpp": mlx_total / n,
        "shard_bpp": (8 * n + 16 * groups) / n,
        "native_bytes": native_total / 8,
        "mlx_bytes": mlx_total / 8,
        "shard_bytes": n + 2 * groups,
    }


# ------------------------------------------------------ shards and checkpoints ---

GROUP_SIZE = 64  # group size of shards that do not store one (older format)


def _module_path(layer: int, short: str) -> str:
    if short not in WKEY:
        raise ValueError(f"unknown linear {short!r} in shards")
    return f"model.layers.{layer}." + WKEY[short][: -len(".weight")]


def load_shards(shard_dir: str) -> dict:
    """layer_XX.npz shards -> {module path: dict(codes, scales, bits, group_size)}.

    Reads the format of gsq.pipeline.save_shard: per linear <short>.codes (uint8), <short>.scales
    (fp16) and <short>.bits, with the group size in meta.fingerprint. Scales come back as float32,
    as the pipeline's own export passes them. Shards with a file-level bits (and optionally grid
    and group_size) are read too.
    """
    quantized = {}
    for fn in sorted(glob.glob(os.path.join(shard_dir, "layer_*.npz"))):
        m = re.fullmatch(r"layer_(\d+)\.npz", os.path.basename(fn))
        if m is None:
            raise ValueError(f"unexpected shard name {fn}")
        layer = int(m.group(1))
        with np.load(fn) as shard:
            files = set(shard.files)
            if "meta" in files:
                group_size = int(json.loads(str(shard["meta"]))["fingerprint"]["group_size"])
            else:
                group_size = int(shard["group_size"]) if "group_size" in files else GROUP_SIZE
            for key in shard.files:
                if not key.endswith(".codes"):
                    continue
                short = key[: -len(".codes")]
                bits = int(shard[f"{short}.bits"]) if f"{short}.bits" in files else int(shard["bits"])
                if "grid" in files and not np.array_equal(shard["grid"], get_grid(bits)):
                    raise ValueError(f"{fn}: stored grid differs from get_grid({bits})")
                quantized[_module_path(layer, short)] = dict(
                    codes=shard[key], scales=shard[f"{short}.scales"].astype(np.float32),
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
    """Plain RTN for every base linear, as native codes and scales (the RTN baseline)."""
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


def export_checkpoint(quantized: dict, out_dir, base_dir) -> dict:
    """Write the mlx-lm checkpoint of the quantized linears (the base's other tensors are copied)
    and return the bit accounting and file sizes of what was written."""
    n_params = sum(q["codes"].size for q in quantized.values())
    export_mlx_lm(out_dir, quantized, base_dir)
    rep = bits_report(quantized, n_params)
    hist = dict(sorted(Counter(q["bits"] for q in quantized.values()).items()))
    return {"dir": str(out_dir), "linears": len(quantized), "weights": int(n_params),
            "bits_histogram": {str(k): v for k, v in hist.items()},
            "bits_per_weight": {k: rep[k] for k in ("code_bpp", "native_bpp", "mlx_bpp", "shard_bpp")},
            "quantized_bytes": rep["mlx_bytes"],
            "safetensors_bytes": os.path.getsize(os.path.join(out_dir, "model.safetensors"))}
