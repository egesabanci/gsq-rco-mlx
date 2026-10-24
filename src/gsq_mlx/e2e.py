"""Phase-2b end-to-end GSQ on real Qwen2.5-0.5B — streaming, low-memory.

Memory design (~70MB free): weights are NEVER fully materialized.
A dependency-free safetensors header parser + os.pread reads each tensor
(most <9MB) on demand as uint16 -> ``mx.array(...).view(mx.bfloat16)``.
(``mx.load`` would materialize all 942MB; lazy reads are the honest
equivalent — same bytes, no torch anywhere.) Peak transient per block is
one linear's Lion chunk (~10-20MB) + hidden states (~4MB).

Pipeline (all batch-1, one sequence per forward):
  1. tokenize repo text (gsq.txt -> 8x256 cal, rco.txt -> 4x256 held-out eval)
  2. FP perplexity on eval seqs (lazy per-block weights, vocab-sharded NLL)
  3. streaming quantize: per block, capture FP linear inputs, RTN/group64
     init, short Lion joint opt of assignments+scales (2-bit full grid;
     3/4-bit local-shift; 8-bit RTN-only), write shard npz, forward
     quantized block (frozen prefix), checkpoint hidden states
  4. quantized/hybrid perplexity from final eval hidden states
  5. assemble results/qwen2.5-0.5b-gsq3bit.npz + results/gsq_eval.json

Uses quant.py primitives (gs_softmax, local_shift_values, gs_schedules),
lion.Lion (lr 1e-4 logits / 2e-5 scales, wd 1.0) and gptq.py RTN/init.
"""
from __future__ import annotations
import json
import os
import struct
import time

import numpy as np

try:
    import mlx.core as mx
    _HAS_MLX = True
except Exception:  # pragma: no cover
    mx = None
    _HAS_MLX = False

from .quant import (get_grid, gs_softmax, gs_schedules, local_shift_values,
                    TAU_START, TAU_END, KAPPA_START, KAPPA_END)
from .lion import Lion, LR_LOGITS, LR_SCALES_QWEN
from .gptq import init_logits_2bit_from_codes, init_shift_logits

LIN_ORDER = ("q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj")
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
    """One-sequence-at-a-time Qwen2 forward. Weights dict maps
    'q_proj'..'down_proj' (+optional 'q_bias'..) to mx [out,in] arrays,
    plus 'in_norm', 'post_norm' vectors. Same code serves FP lazy
    weights and dequantized weights."""

    def __init__(self, n_heads=14, n_kv=2, theta=1000000.0, eps=1e-6):
        self.nh, self.nkv, self.theta, self.eps = n_heads, n_kv, theta, eps
        self.rep = n_heads // n_kv
        self.hd = None

    def block(self, h, W: dict, cos=None, sin=None, capture: bool = False):
        # h: [T,H] bf16
        T, H = h.shape
        if self.hd is None:
            self.hd = W["q_proj"].shape[0] // self.nh
        hn = rmsnorm(h, W["in_norm"], self.eps)
        outs = {}
        if capture:
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
        p = mx.softmax(sc, axis=-1).astype(h.dtype)
        o = (p @ v).transpose(0, 2, 1, 3).reshape(T, self.nh * hd)
        if capture:
            outs["o_proj"] = o
        o = o @ W["o_proj"].T
        h1 = h + o
        mn = rmsnorm(h1, W["post_norm"], self.eps)
        if capture:
            outs["gate_proj"], outs["up_proj"] = mn, mn
        g = mn @ W["gate_proj"].T
        u = mn @ W["up_proj"].T
        d_in = silu(g) * u
        if capture:
            outs["down_proj"] = d_in
        d = d_in @ W["down_proj"].T
        out = h1 + d
        if capture:
            return out, outs
        return out


# ----------------------------------------------- per-linear Lion trainer ---

def _to_np32(a):
    """mx (any dtype incl. bf16) or numpy -> fp32 numpy. bf16 buffers
    are unreadable by numpy, so cast inside mx first."""
    if _HAS_MLX and isinstance(a, mx.array):
        a = a.astype(mx.float32)
        mx.eval(a)
    return np.asarray(a, dtype=np.float32)


def _rtn_fast(Wnp: np.ndarray, grid: np.ndarray, group_size: int = 64):
    """Vectorized RTN-nearest (row-chunked): same math as
    gptq.rtn_quantize_groupwise but without the Python row loop, so big
    MLP linears init in ~0.1s instead of ~10s."""
    Wnp = np.asarray(Wnp, dtype=np.float32)
    g = np.asarray(grid, dtype=np.float32)
    gmax = float(np.abs(g).max())
    out_f, in_f = Wnp.shape
    ng = int(np.ceil(in_f / group_size))
    pad = ng * group_size - in_f
    Wp = np.pad(Wnp, ((0, 0), (0, pad))) if pad else Wnp
    scales = np.zeros((out_f, ng), dtype=np.float32)
    codes = np.zeros((out_f, in_f), dtype=np.int64)
    for r0 in range(0, out_f, 512):
        r1 = min(r0 + 512, out_f)
        G = Wp[r0:r1].reshape(r1 - r0, ng, group_size)
        s = np.abs(G).max(axis=-1) / gmax
        s = np.maximum(s, 1e-8)
        scales[r0:r1] = s
        idx = np.argmin(np.abs((G / s[..., None])[..., None] - g),
                          axis=-1)
        codes[r0:r1] = idx.reshape(r1 - r0, ng * group_size)[:, :in_f]
    wq = g[codes] * scales[:, gid_of(in_f, ng, group_size)]
    return {"scales": scales, "codes": codes, "w_q": wq}


def gid_of(in_f: int, ng: int, group_size: int = 64):
    return np.minimum(np.arange(in_f) // group_size, ng - 1)


def _chunk_rows(in_dim: int, K: int, budget: int = 1_500_000) -> int:
    return max(1, int(budget // max(1, in_dim * K)))


def train_linear_mx(Xs, Wfp, bits: int, steps: int, n_resamples: int,
                    seed: int, max_tokens: int = 512, verbose: bool = False):
    """Lion joint opt of assignments+scales for one FP linear.

    Xs: list of [T,in] bf16 mx arrays; stacked once (single matmul in
    the loss — same math as a token loop, far less graph overhead).
    Only the first ``max_tokens`` tokens (evenly strided) enter the
    Lion loop so big MLP linears stay far under the memory ceiling.
    Wfp: [out,in] mx. Returns (codes uint8 np, scales fp16 np, info).
    2-bit: full-grid logits; 3/4-bit: local-shift; 8-bit: RTN-only."""
    grid = get_grid(bits)
    K = len(grid)
    out_f, in_f = tuple(Wfp.shape)
    Wnp = _to_np32(Wfp)
    Xm = mx.concatenate(Xs, axis=0)  # [M,in] bf16, one stacked matmul
    M = Xm.shape[0]
    if M > max_tokens:
        sel = np.linspace(0, M - 1, max_tokens).astype(np.int64)
        Xm = mx.take(Xm, mx.array(sel), axis=0)
    mx.eval(Xm)
    rtn = _rtn_fast(Wnp, np.asarray(grid, dtype=np.float32), 64)
    j0 = rtn["codes"]
    scales0 = rtn["scales"]  # [out, ng]
    ng = scales0.shape[1]
    gid = (np.arange(in_f) // 64)
    gid = np.minimum(gid, ng - 1)

    if bits == 8:  # RTN-only path; Lion on 256 logits/weight is out of scope
        return (j0.astype(np.uint8), scales0.astype(np.float16),
                {"mode": "rtn-only", "steps": 0})

    chunk = _chunk_rows(in_f, 5 if bits > 2 else K)
    codes = np.zeros_like(j0, dtype=np.uint8)
    scales = np.zeros_like(scales0)
    grid_mx = mx.array(np.asarray(grid, dtype=np.float32))
    Wm = mx.array(Wnp)  # fp32, sliced per chunk below
    hard_mses, steps_done = [], 0

    for r0 in range(0, out_f, chunk):
        r1 = min(r0 + chunk, out_f)
        d = r1 - r0
        j0c = j0[r0:r1].reshape(-1)
        gidc = np.tile(gid, (r1 - r0, 1)).reshape(-1)
        rows = np.repeat(np.arange(r1 - r0), in_f)
        s_idx = (rows * ng + gidc).astype(np.int32)  # flat (row,group) index
        s_init = scales0[r0:r1].reshape(-1).astype(np.float32)
        if bits == 2:
            lg = init_logits_2bit_from_codes(
                j0[r0:r1], K, seed=seed + r0).astype(np.float32)
        else:
            lg = init_shift_logits(d * in_f, seed=seed + r0).astype(np.float32)
        lg_m = mx.array(lg)
        s_m = mx.array(s_init)
        if bits > 2:
            C_m = local_shift_values(mx.array(j0c.astype(np.int32)),
                                     grid_mx)  # [d_flat,5] constant
            mx.eval(C_m)

        opt_l = Lion({"logits": lg_m}, lr=LR_LOGITS)
        opt_s = Lion({"s": s_m}, lr=LR_SCALES_QWEN)
        idx_m = mx.array(s_idx)

        def make_loss(g, tau, kappa):
            Xf = Xm.astype(mx.float32)
            Wc = Wm[r0:r1]
            mx.eval(Xf, Wc)

            def loss_fn(logits, s):
                p = gs_softmax(logits, g, tau, kappa)
                if bits == 2:
                    c = p @ grid_mx
                else:
                    c = mx.sum(p * C_m, axis=-1)
                w = s[idx_m] * c
                R = Xf @ (w.reshape(r1 - r0, in_f) - Wc).T
                return mx.mean(R * R)
            return loss_fn

        for t in range(steps):
            tau, kappa = gs_schedules(t, steps)
            gl_acc, gs_acc, n = None, None, 0
            for r_ in range(n_resamples):
                key = mx.random.key((seed + 1) * 100003 + r0 * 101
                                    + t * 17 + r_ * 131)
                g = mx.random.gumbel(shape=list(lg_m.shape), key=key)
                val, (gl, gs) = mx.value_and_grad(
                    make_loss(g, tau, kappa), argnums=(0, 1))(lg_m, s_m)
                mx.eval(val, gl, gs)
                gl_acc = gl if gl_acc is None else gl_acc + gl
                gs_acc = gs if gs_acc is None else gs_acc + gs
                n += 1
            gl_acc, gs_acc = gl_acc / n, gs_acc / n
            out = opt_l.step({"logits": lg_m}, {"logits": gl_acc})
            lg_m = out["logits"]
            out = opt_s.step({"s": s_m}, {"s": gs_acc})
            s_m = out["s"]
            mx.eval(lg_m, s_m)
        steps_done = steps
        mx.eval(lg_m, s_m)
        lg_np = np.asarray(lg_m, dtype=np.float32)
        s_np = np.asarray(s_m, dtype=np.float32).reshape(r1 - r0, ng)
        if bits == 2:
            idx = lg_np.argmax(axis=-1).reshape(r1 - r0, in_f)
        else:
            dl = lg_np.reshape(-1, 5).argmax(axis=-1) - 2
            idx = np.clip(j0[r0:r1].reshape(-1) + dl, 0, K - 1
                          ).reshape(r1 - r0, in_f)
        codes[r0:r1] = idx.astype(np.uint8)
        scales[r0:r1] = s_np
        # hard recon mse for the log (numpy, chunk-sized)
        wq = (s_np[:, gid] * np.asarray(grid, dtype=np.float32)[idx])
        R = _to_np32(Xm) @ (wq - Wnp[r0:r1]).T
        hard_mses.append(float((R * R).mean()))
        mx.clear_cache()
    info = {"mode": "lion-2bit" if bits == 2 else "lion-shift",
            "steps": steps_done,
            "hard_mse": float(np.mean(hard_mses)) if hard_mses else 0.0}
    return codes, scales.astype(np.float16), info


def dequant_linear(codes_u8, scales_fp16, grid):
    """uint8 codes + per-group fp16 scales -> fp16 weight [out,in]."""
    codes = mx.array(codes_u8.astype(np.int32))
    sc = mx.array(scales_fp16.astype(np.float32))
    g = mx.array(np.asarray(grid, dtype=np.float32))
    out_f, in_f = codes_u8.shape
    ng = sc.shape[1]
    gid = mx.array(np.minimum(np.arange(in_f) // 64, ng - 1))
    s_full = sc[:, gid]
    return (s_full * mx.take(g, codes)).astype(mx.bfloat16)


# ------------------------------------------------------ vocab-sharded NLL ---

def seq_nll(h_last, targets, get_rows, vocab: int, shard: int = 8192,
          get_range=None):
    """NLL of targets[1:] given final-norm hidden states. Online logsumexp
    over vocab shards (single-pread contiguous ``get_range`` when given);
    target logits gathered from cached embed rows."""
    H = h_last[:-1].astype(mx.float32)          # [T-1, E]
    tgt = np.asarray(targets[1:], dtype=np.int64)
    Et = get_rows(tgt)                           # [T-1, E]
    mx.eval(Et)
    t_logit = mx.sum(H * Et.astype(mx.float32), axis=-1)  # [T-1]
    lse = mx.full([H.shape[0]], -1e30, dtype=mx.float32)
    for a in range(0, vocab, shard):
        b = min(a + shard, vocab)
        E = (get_range(a, b) if get_range is not None
             else get_rows(range(a, b))).astype(mx.float32)
        L = H @ E.T
        mx.eval(L)
        m = mx.maximum(lse, mx.max(L, axis=-1))
        lse = m + mx.log(mx.exp(lse - m) + mx.sum(mx.exp(L - m[:, None]),
                                                  axis=-1))
        del E, L
        mx.clear_cache()
    mx.eval(lse, t_logit)
    return float(mx.mean(lse - t_logit)), int(tgt.shape[0])
