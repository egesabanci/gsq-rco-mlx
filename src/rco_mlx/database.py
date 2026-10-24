"""Quantized weight database for the RCO objective (paper App B, Eq.6).

For every linear i and every option k (bit-width b_k) the database stores the quantizer
that deployment uses: gsq_mlx.gptq.quantize_gptq (signed scales, group size 64) by default,
or quantize_rtn with --quantizer rtn. A one-hot assignment assembles linear i as
dequant(db[i, k_i]); the App-B residual is Delta_ik = dequant(db[i, k]) - W_ref_i.

Hessians are the calibration Gram matrices 2 X^T X / M of the FP model's inputs to each
linear, with position 0 of every window excluded (finding #10). They are computed from the
FP model layer by layer, so each layer's inputs are the FP model's (not a quantized prefix).

Layout under <db_dir>:
  manifest.json                            identity of the database (see verify())
  sensitivity.npy                          float64 [n_groups, K]: per-linear output MSE
                                           0.5 tr(dW H dW^T) on the calibration Hessian
  L{layer:02d}.{name}.b{bits}.codes.npy    uint8 [out, in]
  L{layer:02d}.{name}.b{bits}.scales.npy   float32 [out, n_groups]
  .gitignore                               "*": keeps the database out of git
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from gsq_mlx.gptq import hessian_from_inputs, quantize_gptq, quantize_rtn
from gsq_mlx.quant import get_grid
from gsq_rco_common.model import LINEARS, get_linear_weight

FORMAT = "rco_mlx.database/1"
MAX_OPTIONS = 255
HESSIAN_CHUNK_ROWS = 512
# Hessian site per linear: q/k/v read the same normed input, gate/up the same MLP input.
SITE_OF = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "o",
           "gate_proj": "gate_up", "up_proj": "gate_up", "down_proj": "down"}
TAP_OF = {"qkv": ("self_attn", "q_proj"), "o": ("self_attn", "o_proj"),
          "gate_up": ("mlp", "gate_proj"), "down": ("mlp", "down_proj")}
IDENTITY_KEYS = ("format", "model_weights_sha256", "options", "group_size", "quantizer",
                 "damp", "signed_scale", "n_layers", "calibration", "hessian")


class StaleDatabase(RuntimeError):
    """The database was built from other model weights, calibration tokens or settings."""


@dataclass(frozen=True)
class DBSpec:
    options: tuple[int, ...] = (2, 3, 4, 5, 6, 8)
    group_size: int = 64
    quantizer: str = "gptq"          # "gptq" (deployment quantizer) or "rtn"
    damp: float = 0.01               # GPTQ dampening, ignored by rtn
    n_layers: int = 24
    calib_split: str = "train"       # calibration tokens come from WikiText-2 train
    calib_seed: int = 0


def sha256_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha256_files(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
    return h.hexdigest()


def identity(spec: DBSpec, model_sha: str, tokens: np.ndarray) -> dict:
    """The fields that decide whether a stored database is valid for a given run."""
    tokens = np.asarray(tokens)
    return {
        "format": FORMAT,
        "model_weights_sha256": model_sha,
        "options": [int(b) for b in spec.options],
        "group_size": int(spec.group_size),
        "quantizer": spec.quantizer,
        "damp": float(spec.damp) if spec.quantizer == "gptq" else None,
        "signed_scale": True,
        "n_layers": int(spec.n_layers),
        "calibration": {
            "split": spec.calib_split,
            "seed": int(spec.calib_seed),
            "n_windows": int(tokens.shape[0]),
            "window_len": int(tokens.shape[1]),
            "excluded_positions": [0],
            "tokens_sha256": sha256_array(tokens),
            "hessian_rows": int(tokens.shape[0] * (tokens.shape[1] - 1)),
        },
        "hessian": "2/M X^T X in float64 (gsq_mlx.gptq.hessian_from_inputs), FP-model inputs",
    }


def memory_report() -> str:
    """MLX active/cached/peak bytes and the process max RSS, in GB (macOS reports bytes)."""
    gb = 1e9
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (f"MLX active {mx.get_active_memory() / gb:.2f} GB, cache "
            f"{mx.get_cache_memory() / gb:.2f} GB, peak {mx.get_peak_memory() / gb:.2f} GB, "
            f"max RSS {rss / gb:.2f} GB")


def _name(layer: int, name: str, bits: int, kind: str) -> str:
    return f"L{layer:02d}.{name}.b{bits}.{kind}.npy"


def _check_spec(spec: DBSpec, n_model_layers: int) -> tuple[int, ...]:
    opts = tuple(int(b) for b in spec.options)
    if not opts or len(opts) > MAX_OPTIONS:
        raise ValueError(f"need 1..{MAX_OPTIONS} options, got {opts}")
    if any(not 2 <= b <= 8 for b in opts) or list(opts) != sorted(set(opts)):
        raise ValueError(f"options must be strictly increasing bit-widths in 2..8, got {opts}")
    if spec.quantizer not in ("gptq", "rtn"):
        raise ValueError(f"quantizer must be 'gptq' or 'rtn', got {spec.quantizer!r}")
    if not 1 <= spec.n_layers <= n_model_layers:
        raise ValueError(f"n_layers must be in 1..{n_model_layers}, got {spec.n_layers}")
    return opts


class _Tap(nn.Module):
    """Records the input of a linear and delegates to it (no change to the output)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.last = None

    def __call__(self, x):
        self.last = x
        return self.inner(x)


def _install_taps(layer):
    taps = {}
    for site, (parent_name, attr) in TAP_OF.items():
        parent = getattr(layer, parent_name)
        orig = getattr(parent, attr)
        tap = _Tap(orig)
        setattr(parent, attr, tap)
        taps[site] = (parent, attr, orig, tap)
    return taps


def _restore_taps(taps):
    for parent, attr, orig, _ in taps.values():
        setattr(parent, attr, orig)


def build_database(model, tokens, spec: DBSpec, db_dir, model_sha: str,
                   log=print, release: bool = False) -> dict:
    """Quantize every linear of the first spec.n_layers layers at every option.

    tokens: int32 [N, T] calibration windows. For each layer the FP block is run on all
    windows; its linear inputs (position 0 dropped) accumulate the four Hessians. Then each
    linear is quantized with the database quantizer at each option, and the output MSE of
    that quantization on the calibration Hessian is stored as the sensitivity.
    release=True frees every FP weight once it is no longer needed: the token embedding after
    the block inputs exist, then each block after its quantization. With a model loaded
    lazily (mlx_lm load_model(lazy=True)) only one block is resident at a time, which keeps
    the peak at one block plus the Hessians. Use it only in a process that will not run the
    model again.
    Returns the manifest, which is written last (atomically) so an interrupted build leaves
    no manifest behind.
    """
    tokens = np.asarray(tokens, dtype=np.int32)
    if tokens.ndim != 2 or tokens.shape[1] < 2:
        raise ValueError(f"tokens must be [N, T] with T >= 2, got {tokens.shape}")
    opts = _check_spec(spec, len(model.model.layers))
    db = Path(db_dir)
    db.mkdir(parents=True, exist_ok=True)
    (db / ".gitignore").write_text("*\n")
    n_win = tokens.shape[0]
    states = [model.model.embed_tokens(mx.array(tokens[w:w + 1])) for w in range(n_win)]
    mx.eval(states)
    if release:
        model.model.embed_tokens.weight = mx.zeros((1, 1), dtype=mx.bfloat16)
        mx.clear_cache()
    n_groups = spec.n_layers * len(LINEARS)
    sens = np.zeros((n_groups, len(opts)), dtype=np.float64)
    counts = []
    for L in range(spec.n_layers):
        layer = model.model.layers[L]
        mx.eval(layer.parameters())   # a lazily loaded model materialises one block at a time
        taps = _install_taps(layer)
        acc = {site: (None, 0) for site in TAP_OF}
        try:
            for w in range(n_win):
                out = layer(states[w], "causal", None)
                for site, (_, _, _, tap) in taps.items():
                    X = np.asarray(tap.last[0, 1:].astype(mx.float32))
                    for r in range(0, X.shape[0], HESSIAN_CHUNK_ROWS):
                        acc[site] = hessian_from_inputs(X[r:r + HESSIAN_CHUNK_ROWS], *acc[site])
                    tap.last = None
                mx.eval(out)
                states[w] = out
        finally:
            _restore_taps(taps)
        log(f"[build-db] layer {L:2d} Hessians ({n_win * (tokens.shape[1] - 1)} rows): "
            f"{memory_report()}")
        H = {site: acc[site][0] for site in TAP_OF}
        for j, name in enumerate(LINEARS):
            i = L * len(LINEARS) + j
            W = np.asarray(get_linear_weight(model, L, name).astype(mx.float32))
            counts.append(int(W.size))
            Hs = H[SITE_OF[name]]
            for k, b in enumerate(opts):
                if spec.quantizer == "gptq":
                    res = quantize_gptq(W, Hs, b, spec.group_size, spec.damp)
                    sens[i, k] = res["err"]
                else:
                    res = quantize_rtn(W, b, spec.group_size)
                    D = res["w_q"].astype(np.float64) - W.astype(np.float64)
                    sens[i, k] = 0.5 * float(np.sum((D @ Hs) * D))
                np.save(db / _name(L, name, b, "codes"), res["codes"])
                np.save(db / _name(L, name, b, "scales"), res["scales"])
        del H, acc, taps
        if release:
            layer.update(tree_map(lambda p: mx.zeros((1,), dtype=p.dtype), layer.parameters()))
        mx.clear_cache()
        log(f"[build-db] layer {L:2d} done ({len(LINEARS)} linears x {len(opts)} options): "
            f"{memory_report()}")
    np.save(db / "sensitivity.npy", sens)
    manifest = identity(spec, model_sha, tokens)
    manifest.update({
        "counts": counts,
        "groups": [f"{L}.{name}" for L in range(spec.n_layers) for name in LINEARS],
        "sensitivity_file": "sensitivity.npy",
    })
    tmp = db / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    os.replace(tmp, db / "manifest.json")
    return manifest


def dequant_mx(codes: np.ndarray, scales: np.ndarray, bits: int, group_size: int) -> mx.array:
    """w[:, c] = scales[:, c // group_size] * grid[codes[:, c]] on the device, float32.

    Same values as gsq_mlx.gptq.dequantize (checked in scripts/test_rco_alloc.py).
    """
    codes = np.asarray(codes)
    out_f, in_f = codes.shape
    if scales.shape != (out_f, -(-in_f // group_size)):
        raise ValueError(f"scales {scales.shape} do not match codes {codes.shape}")
    grid = mx.array(get_grid(bits).astype(np.float32))
    vals = mx.take(grid, mx.array(codes).astype(mx.int32), axis=0)
    s = mx.repeat(mx.array(np.asarray(scales, dtype=np.float32)), group_size, axis=1)
    return vals * s[:, :in_f]


class Database:
    """Read-only view of a built database. Codes are read per call, never all at once."""

    def __init__(self, db_dir):
        self.dir = Path(db_dir)
        path = self.dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no database manifest at {path}; build it with "
                "scripts/allocate_qwen.py build-db")
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("format") != FORMAT:
            raise StaleDatabase(f"{path}: format {self.manifest.get('format')!r}, expected {FORMAT!r}")
        m = self.manifest
        self.options = tuple(int(b) for b in m["options"])
        self.group_size = int(m["group_size"])
        self.n_layers = int(m["n_layers"])
        self.groups = [(L, name) for L in range(self.n_layers) for name in LINEARS]
        self.counts = np.asarray(m["counts"], dtype=np.int64)
        self.sensitivity = np.load(self.dir / m["sensitivity_file"])
        if self.counts.shape != (len(self.groups),) or \
                self.sensitivity.shape != (len(self.groups), len(self.options)):
            raise StaleDatabase(f"{path}: group/option shapes do not match the manifest")

    def verify(self, model_sha: str, tokens: np.ndarray) -> None:
        """Raise StaleDatabase unless the model weights and calibration tokens still match."""
        stored = {k: self.manifest.get(k) for k in IDENTITY_KEYS}
        now = dict(stored, model_weights_sha256=model_sha)
        now["calibration"] = dict(stored["calibration"], tokens_sha256=sha256_array(tokens),
                                  n_windows=int(np.shape(tokens)[0]),
                                  window_len=int(np.shape(tokens)[1]))
        diffs = [k for k in IDENTITY_KEYS if stored[k] != now[k]]
        if diffs:
            raise StaleDatabase(f"{self.dir} is stale: {', '.join(diffs)} changed; rebuild it")

    def dequant(self, i: int, k: int) -> mx.array:
        """Assembled weight of group i at option k, float32 [out, in]."""
        L, name = self.groups[i]
        bits = self.options[k]
        codes = np.load(self.dir / _name(L, name, bits, "codes"))
        scales = np.load(self.dir / _name(L, name, bits, "scales"))
        return dequant_mx(codes, scales, bits, self.group_size)
