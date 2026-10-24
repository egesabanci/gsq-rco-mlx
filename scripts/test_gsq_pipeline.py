"""WP5 acceptance test: block-wise staged GSQ (gsq_mlx.staged) and the pipeline driver.

  A. toy blocks (random weights, no checkpoint): streaming Hessian equals gptq.hessian_from_inputs,
     the compiled Lion rule equals lion.Lion, Gumbel moments, stage losses use the FP targets and skip
     position 0, the stream applies each block once, every bit-width path (1, 2, 3, 8), the scale-only
     fine-tune lowers the KL (8-bit scales included), shard and allocation I/O round-trip, the
     local-shift soft weight equals the hard decode at a near-zero temperature (3, 5, 6 bits), and
     scale_lr keeps lr_scales at 2 and 3 bits and freezes the group scales above 3 bits
  B. real Qwen2.5-0.5B layer 5 with 16 x 256 WikiText-2 tokens: at the default logit lr (1e-4), 100 of
     the 200 default steps lower every stage's hard objective below the GPTQ init, and resuming a 2-block
     run from its first shard reproduces the second shard exactly. Layer 3 with the RCO allocation's
     8-bit k_proj (the layer where the full run crashed), 3 steps on 4 x 128 tokens: 8-bit linears keep
     their GPTQ codes and scales, also when q_proj is 8 bits too. B4: layer 3 with the RCO bits
     (q4 k8 v6 o3 gate4 up3 down5) at 128 x 512 tokens, 200 steps, on the FP output of layers 0-2:
     every stage's hard objective ends at or below its GPTQ init (HEAD's mlp stage rises here)

Usage: .venv/bin/python scripts/test_gsq_pipeline.py   (run it through heavy.sh: it loads layers 3
       and 5; about 6.5 min, of which B4 is about 3.5 min; max RSS 2.80 GiB). Exits nonzero on any
       failure.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import mlx.core as mx
import numpy as np

from gsq_mlx import staged
from gsq_mlx.e2e import LIN_ORDER, LazyWeights, Qwen2Stream, rmsnorm
from gsq_mlx.gptq import dequantize, hessian_from_inputs, quantize_gptq
from gsq_mlx.lion import Lion, scale_lr
from gsq_mlx.quant import get_grid

T0 = time.time()
MODEL = os.path.join(ROOT, "models", "Qwen2.5-0.5B", "model.safetensors")
spec = importlib.util.spec_from_file_location("quantize_qwen",
                                              os.path.join(ROOT, "scripts", "quantize_qwen.py"))
qq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qq)
quiet = lambda *a, **k: None  # noqa: E731


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        raise SystemExit(f"test_gsq_pipeline FAILED at {name}")


def quantize_recording_inits(stream, h, W, bits, opts, layer):
    """quantize_block, also returning each linear's GPTQ init (codes, fp16 scales) as _Linear got it."""
    inits, orig_init = {}, staged._Linear.__init__

    def record(self, name, b, codes, scales16, group_size, seed):
        orig_init(self, name, b, codes, scales16, group_size, seed)
        inits[name] = (codes.copy(), scales16.copy())

    staged._Linear.__init__ = record
    try:
        shard, stats, _ = staged.quantize_block(stream, h, W, bits, opts, layer, log=quiet)
    finally:
        staged._Linear.__init__ = orig_init
    return shard, stats, inits


def at_gptq_init(shard, inits, names):
    return all(shard[n]["codes"].dtype == np.uint8 and np.array_equal(shard[n]["codes"], inits[n][0])
               and np.array_equal(shard[n]["scales"], inits[n][1]) for n in names)


# ---------------------------------------------------------------- toy model ---
H, I, NH, NKV = 128, 256, 4, 2
KV = NKV * (H // NH)
GS = 64


def toy_block(rng, sc=0.05):
    def w(shape, s=sc):
        return mx.array((rng.standard_normal(shape) * s).astype(np.float32)).astype(mx.bfloat16)
    return {"q_proj": w((H, H)), "k_proj": w((KV, H)), "v_proj": w((KV, H)),
            "o_proj": w((H, H)), "gate_proj": w((I, H)), "up_proj": w((I, H)),
            "down_proj": w((H, I)), "q_bias": w((H,), 0.1), "k_bias": w((KV,), 0.1),
            "v_bias": w((KV,), 0.1), "in_norm": mx.ones((H,)).astype(mx.bfloat16),
            "post_norm": mx.ones((H,)).astype(mx.bfloat16)}


def base_of(W):
    return {k: W[k] for k in W if k not in LIN_ORDER}


def bits_all(b):
    return {n: b for n in LIN_ORDER}


rng = np.random.default_rng(0)
stream = Qwen2Stream(n_heads=NH, n_kv=NKV)

# ------------------------------------------------------------------ A. toy ---
X = rng.standard_normal((300, 96)).astype(np.float32)
H_ref, _ = hessian_from_inputs(X.astype(np.float64))
Hs, n = None, 0
for a in range(0, 300, 70):
    Hs, n = staged.hessian_add(Hs, n, mx.array(X[a:a + 70]))
rel = float(np.max(np.abs(Hs - H_ref)) / np.max(np.abs(H_ref)))
check("A1 streaming Hessian equals gptq.hessian_from_inputs", rel < 1e-5 and n == 300,
      f"rel err {rel:.1e}")

x = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)
rows = np.asarray(staged._rows(mx.array(x)))
check("A1 loss rows skip position 0", rows.shape == (8, 3)
      and np.array_equal(rows, x[:, 1:].reshape(-1, 3)))

lrng = np.random.default_rng(2)
P_ref = {"a": mx.array(lrng.standard_normal((5, 7)).astype(np.float32)),
         "b": mx.array(lrng.standard_normal((3,)).astype(np.float32))}
P_new = dict(P_ref)
M_new = {k: mx.zeros_like(v) for k, v in P_new.items()}
lr_map = {k: mx.array(1e-3) for k in P_new}
opt = Lion(dict(P_ref), lr=1e-3)
worst = 0.0
for _ in range(3):
    G = {k: mx.array(lrng.standard_normal(v.shape).astype(np.float32)) for k, v in P_new.items()}
    P_ref = opt.step(P_ref, G)
    P_new, M_new = staged._lion_update(P_new, M_new, G, lr_map)
    worst = max(worst, max(float(mx.max(mx.abs(P_ref[k] - P_new[k]))) for k in P_ref))
check("A2 compiled Lion rule equals lion.Lion (3 steps)", worst < 1e-6, f"max diff {worst:.1e}")

g = np.asarray(staged._gumbel(mx.random.key(0), (400_000,)))
check("A3 Gumbel(0,1) mean and variance", abs(g.mean() - 0.5772) < 0.01
      and abs(g.var() - np.pi ** 2 / 6) < 0.03, f"mean {g.mean():.4f} var {g.var():.4f}")

# stage losses: FP targets, position 0 excluded, soft weight from the same key
Wt = toy_block(rng)
N, T = 3, 16
h = mx.array((rng.standard_normal((N, T, H)) * 0.5).astype(np.float32)).astype(mx.bfloat16)
mx.eval(h)
gq = quantize_gptq(np.asarray(Wt["q_proj"].astype(mx.float32)), np.eye(H), 2, GS, 0.01, 128)
lin_q = staged._Linear("q_proj", 2, gq["codes"], gq["scales"].astype(np.float16), GS, seed=3)
P = {lin_q.key(p): v for p, v in lin_q.P.items()}
Xq = staged._rows(rmsnorm(h, Wt["in_norm"]))
key, tau, kappa = mx.random.key(11), mx.array(1.0), mx.array(100.0)
Wfm = {"q_proj": Wt["q_proj"].astype(mx.float32)}
loss = float(staged._loss_linear([lin_q], Wfm, P, Xq, key, tau, kappa))
Wb = np.asarray(lin_q.soft(P, mx.random.split(key, 1)[0], tau, kappa))
ref = float(np.mean((np.asarray(Xq) @ (Wb - np.asarray(Wfm["q_proj"])).T) ** 2))
check("A4 qk loss = mean ||X (W_bar - W_fp)^T||^2 over positions 1..", abs(loss - ref) <= 1e-4 * ref,
      f"loss {loss:.6e} ref {ref:.6e}")

HN = rmsnorm(h, Wt["in_norm"])
A_fp = mx.stack([stream.attn_core(HN[s], Wt) @ Wt["o_proj"].T for s in range(N)]).astype(mx.bfloat16)
gv = quantize_gptq(np.asarray(Wt["v_proj"].astype(mx.float32)), np.eye(H), 2, GS, 0.01, 128)
gO = quantize_gptq(np.asarray(Wt["o_proj"].astype(mx.float32)), np.eye(H), 2, GS, 0.01, 128)
lin_v = staged._Linear("v_proj", 2, gv["codes"], gv["scales"].astype(np.float16), GS, seed=4)
lin_o = staged._Linear("o_proj", 2, gO["codes"], gO["scales"].astype(np.float16), GS, seed=5)
Pvo = {}
for lin in (lin_v, lin_o):
    Pvo.update({lin.key(p): v for p, v in lin.P.items()})
key2 = mx.random.key(12)
l_attn = float(staged._loss_attn(stream, Wt, [lin_v, lin_o], Pvo, (HN, A_fp), key2, tau, kappa))
ks = mx.random.split(key2, 2)
Wd = dict(Wt, v_proj=lin_v.soft(Pvo, ks[0], tau, kappa))
Wo = np.asarray(lin_o.soft(Pvo, ks[1], tau, kappa))
diffs = [np.asarray((stream.attn_core(HN[s], Wd) @ mx.array(Wo).T).astype(mx.float32))[1:]
         - np.asarray(A_fp[s].astype(mx.float32))[1:] for s in range(N)]
ref_attn = float(np.mean([np.mean(d * d) for d in diffs]))
check("A4 vo loss = mean ||A_bar - A_fp||^2 over positions 1..", abs(l_attn - ref_attn) <= 1e-3 * ref_attn,
      f"loss {l_attn:.6e} ref {ref_attn:.6e}")
A_pert = np.asarray(A_fp.astype(mx.float32)).copy()
A_pert[:, 0] += 10.0
l_pert = float(staged._loss_attn(stream, Wt, [lin_v, lin_o], Pvo, (HN, mx.array(A_pert)), key2, tau, kappa))
check("A4 vo loss ignores target position 0", abs(l_pert - l_attn) <= 1e-5 * max(l_attn, 1e-12),
      f"delta {abs(l_pert - l_attn):.1e}")

Y_fp = mx.stack([stream.block(h[s], Wt) for s in range(N)]).astype(mx.bfloat16)
lins_m = []
for name in ("gate_proj", "up_proj", "down_proj"):
    gm = quantize_gptq(np.asarray(Wt[name].astype(mx.float32)), np.eye(Wt[name].shape[1]), 2, GS, 0.01, 128)
    lins_m.append(staged._Linear(name, 2, gm["codes"], gm["scales"].astype(np.float16), GS, seed=6))
Pm = {}
for lin in lins_m:
    Pm.update({lin.key(p): v for p, v in lin.P.items()})
key3 = mx.random.key(13)
l_blk = float(staged._loss_block(stream, Wt, lins_m, Pm, (h, Y_fp), key3, tau, kappa))
ks3 = mx.random.split(key3, 3)
Wd3 = dict(Wt, **{lin.name: lin.soft(Pm, ks3[i], tau, kappa) for i, lin in enumerate(lins_m)})
ref_blk = float(np.mean([np.mean(np.asarray((stream.block(h[s], Wd3) - Y_fp[s])[1:].astype(mx.float32)) ** 2)
                         for s in range(N)]))
check("A4 mlp loss = mean ||Y_bar - Y_fp||^2 over positions 1..", abs(l_blk - ref_blk) <= 1e-3 * ref_blk,
      f"loss {l_blk:.6e} ref {ref_blk:.6e}")
Y_pert = np.asarray(Y_fp.astype(mx.float32)).copy()
Y_pert[:, 0] += 10.0
l_blk_p = float(staged._loss_block(stream, Wt, lins_m, Pm, (h, mx.array(Y_pert)), key3, tau, kappa))
check("A4 mlp loss ignores target position 0", abs(l_blk_p - l_blk) <= 1e-5 * max(l_blk, 1e-12),
      f"delta {abs(l_blk_p - l_blk):.1e}")

# stream bookkeeping: each block applied once; the second block sees the quantized prefix
opts_toy = staged.BlockOpts(group_size=GS, steps=5, micro_batch=2, resamples=2)
W1 = toy_block(rng)
shard0, _, Wq0 = staged.quantize_block(stream, h, Wt, bits_all(2), opts_toy, layer=0, log=quiet)
h1 = staged.advance(stream, h, Wq0)
shard1, _, Wq1 = staged.quantize_block(stream, h1, W1, bits_all(2), opts_toy, layer=1, log=quiet)
h2 = staged.advance(stream, h1, Wq1)
ref_chain = np.stack([np.asarray(stream.block(stream.block(h[s], Wq0), Wq1).astype(mx.float32))
                      for s in range(N)])
wrong = np.stack([np.asarray(stream.block(stream.block(stream.block(h[s], Wq0), Wq0), Wq1).astype(mx.float32))
                  for s in range(N)])
d_ok = float(np.max(np.abs(np.asarray(h2.astype(mx.float32)) - ref_chain)))
d_bad = float(np.max(np.abs(ref_chain - wrong)))
check("A5 stream applies each block exactly once", d_ok == 0.0 and d_bad > 1e-3,
      f"pipeline vs reference {d_ok:.1e}; applying block 0 twice differs by {d_bad:.1e}")
rebuilt = staged.hard_block(shard0, base_of(Wt), GS)
same = all(np.array_equal(np.asarray(rebuilt[n].astype(mx.float32)), np.asarray(Wq0[n].astype(mx.float32)))
           for n in LIN_ORDER)
check("A5 hard weights equal the stored shard (codes + fp16 scales)", same)

# every bit-width path
for b in (2, 3, 1, 8):
    lo_hi = 3 if b == 1 else 2 ** b
    sh, st, Wq = staged.quantize_block(
        stream, h, Wt, bits_all(b),
        staged.BlockOpts(group_size=GS, steps=60, micro_batch=2, resamples=2, lr_logits=1e-3),
        layer=0, log=quiet)
    codes_ok = all(sh[n]["codes"].max() < lo_hi and sh[n]["codes"].dtype == np.uint8 for n in LIN_ORDER)
    rebuilt = staged.hard_block(sh, base_of(Wt), GS)
    same = all(np.array_equal(np.asarray(rebuilt[n].astype(mx.float32)), np.asarray(Wq[n].astype(mx.float32)))
               for n in LIN_ORDER)
    flips = sum(st["linears"][n]["flip_frac"] for n in LIN_ORDER if "flip_frac" in st["linears"][n])
    objs = {s: (st["stages"][s]["objective_init"], st["stages"][s]["objective_final"]) for s in st["stages"]}
    finite = all(np.isfinite(v) for pair in objs.values() for v in pair)
    if b == 8:
        check("A6 8-bit is GPTQ only (no flips, no training)", flips == 0 and codes_ok and same
              and all(st["stages"][s]["steps"] == 0 for s in st["stages"]))
    else:
        check(f"A6 {b}-bit path: codes in range, stream equals stored form, objectives finite",
              codes_ok and same and finite, f"flip fraction sum {flips:.3f}")

# an 8-bit linear has no logits, and its soft weight is the dequantized GPTQ weight, exactly
g8 = quantize_gptq(np.asarray(Wt["q_proj"].astype(mx.float32)), np.eye(H), 8, GS, 0.01, 128)
s8 = g8["scales"].astype(np.float16)
lin8 = staged._Linear("q_proj", 8, g8["codes"], s8, GS, seed=7)
w8_soft = np.asarray(lin8.soft({}, mx.random.key(1), mx.array(1.0), mx.array(100.0)))
check("A6 8-bit linear: no logits, soft weight equals the dequantized GPTQ weight",
      set(lin8.P) == {"scales"} and np.array_equal(w8_soft, dequantize(g8["codes"], s8.astype(np.float32), 8, GS)))

# 3..7-bit local shift: at a near-zero temperature the soft weight is the hard decode, exactly, with
# random logits (so codes move, and the grid ends clip). Own RNG: the draws above are unchanged.
rng_ls = np.random.default_rng(3)
for b in (3, 5, 6):
    g = quantize_gptq(np.asarray(Wt["q_proj"].astype(mx.float32)), np.eye(H), b, GS, 0.01, 128)
    s16 = g["scales"].astype(np.float16)
    lin = staged._Linear("q_proj", b, g["codes"], s16, GS, seed=3)
    Pl = {"scales": lin.P["scales"],
          "logits": mx.array(rng_ls.standard_normal(lin.P["logits"].shape).astype(np.float32))}
    soft_w = np.asarray(lin.soft({lin.key(p): v for p, v in Pl.items()}, mx.random.key(0),
                                 mx.array(1e-3), mx.array(1e6)))
    hard_codes = lin.hard_codes(Pl)
    check(f"A9 {b}-bit local shift: soft weight at tau 1e-3 equals the hard decode",
          np.array_equal(soft_w, dequantize(hard_codes, s16.astype(np.float32), b, GS)),
          f"codes moved {np.mean(hard_codes != g['codes']):.2f}")

# group-scale step of the stage optimiser (WP5c rule): 2 and 3 bits keep lr_scales, above 3 bits it is 0
check("A10 scale_lr keeps lr_scales at 2 and 3 bits, freezes the group scales above 3 bits",
      scale_lr(5e-5, 2) == 5e-5 and scale_lr(5e-5, 3) == 5e-5
      and all(scale_lr(5e-5, b) == 0.0 for b in (4, 5, 6, 8)))

# 8-bit k in qk, v in vo and gate in mlp, next to trained linears: every stage runs, the 8-bit ones
# keep their GPTQ codes and scales
mixed = dict(bits_all(2), k_proj=8, v_proj=8, gate_proj=8)
sh_m, st_m, in_m = quantize_recording_inits(stream, h, Wt, mixed, opts_toy, layer=0)
check("A6 8-bit k, v and gate in their stages: each stage trains its other linears, 8-bit at GPTQ",
      at_gptq_init(sh_m, in_m, ["k_proj", "v_proj", "gate_proj"])
      and all(st_m["stages"][s]["steps"] == 5 for s in ("qk", "vo", "mlp")))

# shard I/O and allocation parsing
with tempfile.TemporaryDirectory(prefix="wp5_test_") as tmp:
    fp = qq.fingerprint(qq.build_parser().parse_args(["--mode", "real", "--uniform-bits", "2"]))
    path = os.path.join(tmp, "layer_00.npz")
    qq.save_shard(path, 0, sh, bits_all(8), fp, {"x": 1.5, "stages": {}})
    meta, stats, lins = qq.load_shard(path)
    ok = (meta["layer"] == 0 and meta["fingerprint"] == fp and stats["x"] == 1.5
          and all(np.array_equal(lins[n]["codes"], sh[n]["codes"]) and lins[n]["bits"] == 8
                  for n in LIN_ORDER))
    check("A7 shard round trip keeps codes, bits, stats and fingerprint", ok)

    alloc_layer = os.path.join(tmp, "alloc_layer.json")
    json.dump({"per_layer_bits": {"0": 2, "1": 3}}, open(alloc_layer, "w"))
    alloc_lin = os.path.join(tmp, "alloc_lin.json")
    json.dump({"per_linear_bits": {f"0.{n}": (3 if n == "down_proj" else 2) for n in LIN_ORDER}},
              open(alloc_lin, "w"))
    a1 = qq.load_bits(qq.build_parser().parse_args(["--alloc", alloc_layer, "--layers", "0-1"]))
    a2 = qq.load_bits(qq.build_parser().parse_args(["--alloc", alloc_lin, "--layers", "0-0"]))
    bad = False
    try:
        qq.load_bits(qq.build_parser().parse_args(["--alloc", alloc_lin, "--layers", "0-1"]))
    except SystemExit:
        bad = True
    check("A7 allocation parses per-layer and per-linear JSON; missing layers rejected",
          a1[1]["q_proj"] == 3 and a1[0]["down_proj"] == 2 and a2[0]["down_proj"] == 3 and bad)

# scale-only fine-tune (App H) on a 2-block toy model, codes fixed
blocks_fp = [Wt, W1]
qblocks = {0: shard0, 1: shard1}
V = 200
E = mx.array((rng.standard_normal((V, H)) * 0.1).astype(np.float32)).astype(mx.bfloat16)
fnorm = mx.ones((H,)).astype(mx.bfloat16)
tokens = rng.integers(0, V, size=(4, 32)).astype(np.int32)
hf = mx.stack([mx.take(E, mx.array(t), axis=0) for t in tokens])
for Wb_ in blocks_fp:
    hf = staged.advance(stream, hf, Wb_)
h_ref = rmsnorm(hf, fnorm)
mx.eval(h_ref)
scales, kls = staged.scale_finetune(stream, qblocks, {0: base_of(Wt), 1: base_of(W1)}, E, fnorm,
                                    tokens, h_ref, steps=40, lr=5e-5, group_size=GS, log=quiet)
first, last = float(np.mean(kls[:10])), float(np.mean(kls[-10:]))
moved = max(float(np.max(np.abs(scales[0][n].astype(np.float32)
                                - shard0[n]["scales"].astype(np.float32)))) for n in LIN_ORDER)
check("A8 scale-only fine-tune lowers the KL with codes fixed", last < first and moved > 0,
      f"KL {first:.5f} -> {last:.5f}, max scale move {moved:.2e}")
# App H fine-tunes every scale, the 8-bit k_proj of block 1 included
shard1k, _, _ = staged.quantize_block(stream, h1, W1, dict(bits_all(2), k_proj=8), opts_toy,
                                      layer=1, log=quiet)
sc_k, _ = staged.scale_finetune(stream, {0: shard0, 1: shard1k}, {0: base_of(Wt), 1: base_of(W1)},
                                E, fnorm, tokens, h_ref, steps=10, lr=5e-5, group_size=GS, log=quiet)
moved_k = {n: float(np.max(np.abs(sc_k[1][n].astype(np.float32)
                                  - shard1k[n]["scales"].astype(np.float32)))) for n in LIN_ORDER}
check("A8 App H: every linear's scales move, the 8-bit k_proj included",
      min(moved_k.values()) > 0,
      f"min move {min(moved_k.values()):.2e}, k_proj {moved_k['k_proj']:.2e}")

# ------------------------------------------------------- B. real layer 5 ---
from gsq_rco_common.data import calibration_windows  # noqa: E402  (data cache is local)

lw = LazyWeights(MODEL)
W5 = qq.fp_block_weights(lw, 5)
cal = calibration_windows(16, 256, seed=0)
h5 = mx.stack([lw.load_rows("model.embed_tokens.weight", list(s)) for s in cal])
mx.eval(h5)
opts5 = staged.BlockOpts(group_size=GS, steps=100, micro_batch=2, resamples=2)
sh5, st5, _ = staged.quantize_block(Qwen2Stream(), h5, W5, bits_all(2), opts5, layer=5,
                                    log=lambda *a: print("   ", *a, flush=True))
for s in ("qk", "vo", "mlp"):
    o0, o1 = st5["stages"][s]["objective_init"], st5["stages"][s]["objective_final"]
    check(f"B1 layer 5 {s}: hard objective below GPTQ init", o1 < o0, f"{o0:.4e} -> {o1:.4e}")
flips5 = sum(st5["linears"][n]["flip_frac"] for n in LIN_ORDER)
check("B1 layer 5: codes changed by training", flips5 > 0,
      f"flip fraction per linear {[round(st5['linears'][n]['flip_frac'], 4) for n in LIN_ORDER]}")

# layer 3 of the RCO allocation has an 8-bit k_proj in a mixed stage (where the full run crashed)
W3 = qq.fp_block_weights(lw, 3)
h3 = mx.stack([lw.load_rows("model.embed_tokens.weight", list(s))
               for s in calibration_windows(4, 128, seed=0)])
mx.eval(h3)
opts3 = staged.BlockOpts(group_size=GS, steps=3, micro_batch=2, resamples=1)
bits3 = {"q_proj": 4, "k_proj": 8, "v_proj": 6, "o_proj": 3, "gate_proj": 4, "up_proj": 3, "down_proj": 5}
sh3, st3, inits3 = quantize_recording_inits(Qwen2Stream(), h3, W3, bits3, opts3, layer=3)
check("B3 layer 3 with 8-bit k_proj: runs, k_proj keeps its GPTQ codes and scales",
      at_gptq_init(sh3, inits3, ["k_proj"]) and st3["stages"]["qk"]["steps"] == 3
      and st3["linears"]["k_proj"]["flip_frac"] == 0)
sh4, st4, inits4 = quantize_recording_inits(Qwen2Stream(), h3, W3, dict(bits3, q_proj=8), opts3,
                                            layer=3)
qk4 = st4["stages"]["qk"]
check("B3 q_proj and k_proj both 8 bits: qk runs no steps, keeps its objective, both at GPTQ",
      at_gptq_init(sh4, inits4, ["q_proj", "k_proj"]) and qk4["steps"] == 0
      and qk4["objective_final"] == qk4["objective_init"] and st4["stages"]["vo"]["steps"] == 3,
      f"qk objective {qk4['objective_init']:.4e} -> {qk4['objective_final']:.4e}")
del W3, h3, sh3, sh4

# B4: layer 3 with the RCO allocation's bits at the production calibration (128 x 512) and 200 steps,
# on the FP output of layers 0-2 (the quantized prefix of the full run is not in the repo). On this
# input HEAD's mlp stage rose 1.78e-3 -> 1.03e-2 (WP5c; the full run's quantized prefix gives
# 2.47e-3 -> 9.82e-3). With the group scales above 3 bits frozen, every stage ends below its GPTQ init.
h_b4 = mx.stack([lw.load_rows("model.embed_tokens.weight", list(s))
                 for s in calibration_windows(128, 512, seed=0)])
for i in range(3):
    h_b4 = staged.advance(Qwen2Stream(), h_b4, qq.fp_block_weights(lw, i))
mx.eval(h_b4)
bits_rco3 = {"q_proj": 4, "k_proj": 8, "v_proj": 6, "o_proj": 3, "gate_proj": 4, "up_proj": 3,
             "down_proj": 5}
_, st_b4, _ = staged.quantize_block(Qwen2Stream(), h_b4, qq.fp_block_weights(lw, 3), bits_rco3,
                                    staged.BlockOpts(group_size=GS, steps=200, micro_batch=2,
                                                     resamples=2),
                                    layer=3, log=lambda *a: print("   ", *a, flush=True))
for s in ("qk", "vo", "mlp"):
    o0, o1 = st_b4["stages"][s]["objective_init"], st_b4["stages"][s]["objective_final"]
    check(f"B4 layer 3 RCO bits: {s} hard objective at or below GPTQ init", o1 <= o0,
          f"{o0:.4e} -> {o1:.4e}")
del h_b4

with tempfile.TemporaryDirectory(prefix="wp5_resume_") as tmp:
    dirA, dirB = os.path.join(tmp, "A"), os.path.join(tmp, "B")
    run = ["--mode", "real", "--layers", "0-1", "--uniform-bits", "2", "--steps", "10",
           "--n-cal", "4", "--seq-len", "256", "--sanity-windows", "0", "--no-export"]
    qq.run_real(qq.build_parser().parse_args(run + ["--out", dirA]))
    os.makedirs(dirB)
    shutil.copy(os.path.join(dirA, "layer_00.npz"), os.path.join(dirB, "layer_00.npz"))
    qq.run_real(qq.build_parser().parse_args(run + ["--out", dirB, "--resume"]))
    _, _, la = qq.load_shard(os.path.join(dirA, "layer_01.npz"))
    _, _, lb = qq.load_shard(os.path.join(dirB, "layer_01.npz"))
    same_codes = all(np.array_equal(la[n]["codes"], lb[n]["codes"]) for n in LIN_ORDER)
    same_scales = all(np.array_equal(la[n]["scales"], lb[n]["scales"]) for n in LIN_ORDER)
    check("B2 resume: block 1 replayed on the shard of block 0 reproduces the fresh run",
          same_codes and same_scales)
    check("B2 run_log.json written with per-layer stats", os.path.exists(os.path.join(dirB, "run_log.json")))
    bad = False
    try:
        qq.run_real(qq.build_parser().parse_args(run[:-1] + ["--out", dirB, "--resume", "--steps", "11"]))
    except SystemExit:
        bad = True
    check("B2 resume refuses shards made with other settings", bad)

print(f"GSQ PIPELINE TEST OK ({time.time() - T0:.1f}s)", flush=True)
