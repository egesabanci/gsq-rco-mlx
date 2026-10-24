"""Qwen2 building blocks for the GSQ pipeline: lazy safetensors reader and a
one-sequence Qwen2 decoder stream (verified against mlx-lm by
tests/test_forward_reference.py).

Memory design: weights are never fully materialized. A dependency-free
safetensors header parser + os.pread reads each tensor on demand as uint16,
then ``mx.array(...).view(mx.bfloat16)``. (``mx.load`` would materialize the
whole checkpoint; lazy reads are the same bytes without torch.)

The block quantizer (gsq_rco.gsq.staged) and the FP reference both run on
Qwen2Stream. Its weights dict maps 'q_proj'..'down_proj' (+ 'q_bias'..'v_bias')
to [out, in] arrays and holds 'in_norm' / 'post_norm'. The same code runs on
FP weights, dequantized hard weights and fp32 soft training weights.
"""
from __future__ import annotations
import json
import os
import struct

import mlx.core as mx
import numpy as np

from gsq_rco.common.model import LINEARS

LIN_ORDER = LINEARS  # the seven linears of a block, in quantization order
WKEY = {"q_proj": "self_attn.q_proj.weight",
        "k_proj": "self_attn.k_proj.weight",
        "v_proj": "self_attn.v_proj.weight",
        "o_proj": "self_attn.o_proj.weight",
        "gate_proj": "mlp.gate_proj.weight",
        "up_proj": "mlp.up_proj.weight",
        "down_proj": "mlp.down_proj.weight"}
BKEY = {"q_bias": "self_attn.q_proj.bias",
        "k_bias": "self_attn.k_proj.bias",
        "v_bias": "self_attn.v_proj.bias"}


# ------------------------------------------------------------- lazy store ---

class LazyWeights:
    """Minimal safetensors reader: header parse + pread. No torch, no
    full-file materialization; tensors come back as mx bf16 arrays."""

    def __init__(self, path: str):
        self.fd = os.open(path, os.O_RDONLY)
        n = struct.unpack("<Q", os.read(self.fd, 8))[0]
        raw = os.read(self.fd, n)
        self.hdr = json.loads(raw)
        self.base = 8 + n
        self.path = path

    def _bytes(self, name: str):
        e = self.hdr[name]
        assert e["dtype"] == "BF16", e["dtype"]
        s, t = e["data_offsets"]
        buf = os.pread(self.fd, t - s, self.base + s)
        u = np.frombuffer(buf, dtype=np.uint16).reshape(tuple(e["shape"]))
        return u.copy(), tuple(e["shape"])

    def load(self, name: str):
        u, _ = self._bytes(name)
        return mx.array(u).view(mx.bfloat16)

    def load_rows(self, name: str, ids):
        """Gather arbitrary rows (embedding lookup) with one pread/row."""
        e = self.hdr[name]
        rows, cols = e["shape"]
        start = self.base + e["data_offsets"][0]
        rb = cols * 2
        out = np.empty((len(ids), cols), dtype=np.uint16)
        for j, i in enumerate(ids):
            buf = os.pread(self.fd, rb, start + int(i) * rb)
            out[j] = np.frombuffer(buf, dtype=np.uint16)
        return mx.array(out).view(mx.bfloat16)

    def load_range(self, name: str, a: int, b: int):
        e = self.hdr[name]
        rows, cols = e["shape"]
        start = self.base + e["data_offsets"][0]
        buf = os.pread(self.fd, (b - a) * cols * 2, start + a * cols * 2)
        u = np.frombuffer(buf, dtype=np.uint16).reshape((b - a, cols)).copy()
        return mx.array(u).view(mx.bfloat16)

    def close(self):
        os.close(self.fd)


# -------------------------------------------------------------- qwen2 fwd ---

def rmsnorm(x, w, eps: float = 1e-6):
    xf = x.astype(mx.float32)
    v = mx.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * mx.rsqrt(v + eps) * w.astype(mx.float32)).astype(x.dtype)


def rope_cos_sin(t: int, head_dim: int, theta: float):
    freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2,
                                     dtype=mx.float32) / head_dim))
    ang = mx.arange(t, dtype=mx.float32)[:, None] * freq[None, :]
    return mx.cos(ang), mx.sin(ang)  # [T, hd/2] each


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c = cos.reshape(1, 1, -1, h)
    s = sin.reshape(1, 1, -1, h)
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def silu(x):
    return x * mx.sigmoid(x.astype(mx.float32)).astype(x.dtype)


class Qwen2Stream:
    """One-sequence-at-a-time Qwen2 decoder. ``h`` is [T, H]."""

    def __init__(self, n_heads=14, n_kv=2, theta=1000000.0, eps=1e-6):
        self.nh, self.nkv, self.theta, self.eps = n_heads, n_kv, theta, eps
        self.rep = n_heads // n_kv
        self.hd = None

    def attn_core(self, hn, W: dict, cos=None, sin=None, outs: dict | None = None):
        """Causal self-attention before o_proj: normed input hn [T, H] ->
        [T, n_heads * head_dim]. With ``outs``, records the input of q/k/v
        (hn) and of o_proj (this output)."""
        T, H = hn.shape
        if self.hd is None:
            self.hd = W["q_proj"].shape[0] // self.nh
        if outs is not None:
            outs["q_proj"], outs["k_proj"], outs["v_proj"] = hn, hn, hn
        q = hn @ W["q_proj"].T
        k = hn @ W["k_proj"].T
        v = hn @ W["v_proj"].T
        if "q_bias" in W:
            q = q + W["q_bias"]
            k = k + W["k_bias"]
            v = v + W["v_bias"]
        hd = self.hd
        q = q.reshape(T, self.nh, hd).transpose(1, 0, 2)[None]   # [1,H,T,hd]
        k = k.reshape(T, self.nkv, hd).transpose(1, 0, 2)[None]
        v = v.reshape(T, self.nkv, hd).transpose(1, 0, 2)[None]
        if cos is None:
            cos, sin = rope_cos_sin(T, hd, self.theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.rep > 1:
            k = mx.repeat(k, self.rep, axis=1)
            v = mx.repeat(v, self.rep, axis=1)
        sc = (q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2))
        sc = sc / (hd ** 0.5)
        pos = mx.arange(T)
        causal = pos[None, :] <= pos[:, None]  # [T,T]
        sc = sc + mx.where(causal.reshape(1, 1, T, T), 0.0, -1e9)
        p = mx.softmax(sc, axis=-1).astype(hn.dtype)
        o = (p @ v).transpose(0, 2, 1, 3).reshape(T, self.nh * hd)
        if outs is not None:
            outs["o_proj"] = o
        return o

    def block(self, h, W: dict, cos=None, sin=None, capture: bool = False):
        """Decoder block on h [T, H]. With ``capture``, also returns the input
        of every linear (the calibration activations GPTQ needs)."""
        outs = {} if capture else None
        hn = rmsnorm(h, W["in_norm"], self.eps)
        o = self.attn_core(hn, W, cos, sin, outs)
        h1 = h + o @ W["o_proj"].T
        mn = rmsnorm(h1, W["post_norm"], self.eps)
        if capture:
            outs["gate_proj"], outs["up_proj"] = mn, mn
        d_in = silu(mn @ W["gate_proj"].T) * (mn @ W["up_proj"].T)
        if capture:
            outs["down_proj"] = d_in
        out = h1 + d_in @ W["down_proj"].T
        return (out, outs) if capture else out
