"""WP2 quantizer acceptance test: signed-scale RTN, GPTQ, and the shared contract.

  (a) signed-scale RTN beats unsigned RTN on a positive-skewed max (paper Sec 3.2)
  (b) GPTQ output error ||X (W_q - W)^T||^2 below RTN at 2/3/4 bits, correlated X
  (c) real Qwen2.5-0.5B layer 5 (q_proj, down_proj): GPTQ vs RTN output error on the
      calibration rows (the GPTQ objective) and on a held-out sequence
  (d) dequantize(quantize_*) == w_q, codes in range, ragged group, ternary path,
      dead inputs, block-size invariance, H = I equals RTN, streaming H, err formula

Usage: .venv/bin/python scripts/test_gptq.py   (< 90 s; loads one layer lazily)
Exit nonzero on any failure.
"""
from __future__ import annotations
import gc
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np

from gsq_mlx.gptq import (_tri_inv_lower, dequantize, hessian_from_inputs,
                          quantize_gptq, quantize_rtn)
from gsq_mlx.quant import get_grid

T0 = time.time()


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        raise SystemExit(f"test_gptq FAILED at {name}")


def out_err(X: np.ndarray, Wq: np.ndarray, W: np.ndarray) -> float:
    """||X (W_q - W)^T||^2 / M: mean squared calibration output error."""
    D = X @ (Wq.astype(np.float64) - W.astype(np.float64)).T
    return float((D * D).sum() / X.shape[0])


def scale_per_weight(scales: np.ndarray, in_f: int, gs: int) -> np.ndarray:
    return np.repeat(scales.astype(np.float64), gs, axis=1)[:, :in_f]


# ---- (a) signed vs unsigned RTN ---------------------------------------------
rng = np.random.default_rng(0)
Wa = rng.standard_normal((64, 256)) * 0.02
Wa[:, ::64] = 0.1  # one large positive weight per group: positive-skewed max
e_uns = float(((quantize_rtn(Wa, 2, 64, signed_scale=False)["w_q"] - Wa) ** 2).sum())
sig = quantize_rtn(Wa, 2, 64, signed_scale=True)
e_sig = float(((sig["w_q"] - Wa) ** 2).sum())
check("(a) signed RTN error strictly below unsigned",
      e_sig < e_uns,
      f"unsigned={e_uns:.4e} signed={e_sig:.4e} ratio={e_sig / e_uns:.3f} "
      f"negative-scale groups={float((sig['scales'] < 0).mean()):.2f}")

# ---- (b) GPTQ vs RTN on correlated synthetic inputs -------------------------
rng = np.random.default_rng(1)
Z = rng.standard_normal((2048, 128))
Xc = Z @ (np.eye(128) + 0.25 * rng.standard_normal((128, 128))).T
Hc, nc = hessian_from_inputs(Xc)
Wb = rng.standard_normal((32, 128)) * 0.02
for bits in (2, 3, 4):
    r = quantize_rtn(Wb, bits, 32)
    g = quantize_gptq(Wb, Hc, bits, 32)
    e_r, e_g = out_err(Xc, r["w_q"], Wb), out_err(Xc, g["w_q"], Wb)
    check(f"(b) GPTQ output error below RTN at {bits}-bit", e_g < e_r,
          f"rtn={e_r:.4e} gptq={e_g:.4e} ratio={e_g / e_r:.3f}")
    check(f"(b) reported err equals ||X dW^T||^2/M at {bits}-bit",
          abs(g["err"] - e_g) <= 1e-8 * max(e_g, 1e-30) + 1e-14,
          f"err={g['err']:.6e} direct={e_g:.6e}")

# ---- (c) real Qwen2.5-0.5B layer 5 ------------------------------------------
import mlx.core as mx
from tokenizers import Tokenizer
from gsq_mlx.e2e import BKEY, WKEY, LazyWeights, Qwen2Stream

MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
GS = 128  # paper Tables 7/9 group size; 896 and 4864 are multiples of 128
N_SEQ, SEQ_LEN, LAYER = 5, 256, 5  # seqs 0-3 calibrate H, seq 4 is held out

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
with open(os.path.join(ROOT, "gsq.txt"), "rb") as f:
    text = f.read().replace(b"\x00", b"").decode("utf-8", errors="replace")
ids = tok.encode(text).ids
assert len(ids) >= N_SEQ * SEQ_LEN, "gsq.txt too short for the calibration tokens"
seqs = [ids[k * SEQ_LEN:(k + 1) * SEQ_LEN] for k in range(N_SEQ)]

lw = LazyWeights(os.path.join(MODEL_DIR, "model.safetensors"))


def load_block(i: int) -> dict:
    P = f"model.layers.{i}."
    W = {short: lw.load(P + full) for short, full in WKEY.items()}
    W.update({short: lw.load(P + full) for short, full in BKEY.items()})
    W["in_norm"] = lw.load(P + "input_layernorm.weight")
    W["post_norm"] = lw.load(P + "post_attention_layernorm.weight")
    mx.eval(*W.values())
    return W


blocks = [load_block(i) for i in range(LAYER + 1)]
stream = Qwen2Stream()
cap = {"q_proj": [], "down_proj": []}
for seq in seqs:
    h = lw.load_rows("model.embed_tokens.weight", seq)
    for i in range(LAYER):
        h = stream.block(h, blocks[i])
    _, c = stream.block(h, blocks[LAYER], capture=True)
    for name in cap:
        cap[name].append(np.asarray(c[name][1:].astype(mx.float32)))  # drop position 0
lw.close()
weights = {name: np.asarray(blocks[LAYER][name].astype(mx.float32), dtype=np.float64)
           for name in ("q_proj", "down_proj")}
del blocks, stream  # GPTQ on down_proj needs ~1 GB; release the decoder blocks first
mx.clear_cache()
gc.collect()

for name in ("q_proj", "down_proj"):
    Wl = weights.pop(name)
    Xs = cap.pop(name)
    H_real, n_real = None, 0
    for X in Xs[:4]:
        H_real, n_real = hessian_from_inputs(X, H_real, n_real)
    Xcal = np.concatenate(Xs[:4], axis=0)
    Xhold = Xs[4]
    del Xs
    ref_cal = float(((Xcal @ Wl.T) ** 2).sum() / Xcal.shape[0])
    ref_hold = float(((Xhold @ Wl.T) ** 2).sum() / Xhold.shape[0])
    for bits in (2, 3, 4):
        r = quantize_rtn(Wl, bits, GS)
        g = quantize_gptq(Wl, H_real, bits, GS)
        e_rc, e_gc = out_err(Xcal, r["w_q"], Wl), out_err(Xcal, g["w_q"], Wl)
        e_rh, e_gh = out_err(Xhold, r["w_q"], Wl), out_err(Xhold, g["w_q"], Wl)
        print(f"      layer{LAYER} {name} {bits}-bit relative output error: "
              f"calib rtn={e_rc / ref_cal:.4f} gptq={e_gc / ref_cal:.4f} | "
              f"held-out rtn={e_rh / ref_hold:.4f} gptq={e_gh / ref_hold:.4f}")
        check(f"(c) layer{LAYER} {name} {bits}-bit: GPTQ below RTN on calibration rows",
              e_gc < e_rc, f"ratio={e_gc / e_rc:.3f}")
        check(f"(c) layer{LAYER} {name} {bits}-bit: GPTQ below RTN on held-out seq",
              e_gh < e_rh, f"ratio={e_gh / e_rh:.3f}")

# ---- (d) contract, round trips, and invariants ------------------------------
rng = np.random.default_rng(2)
Wd = rng.standard_normal((6, 100)) * 0.05
for bits in (1, 2, 3, 4):
    grid = get_grid(bits)
    r = quantize_rtn(Wd, bits, 64)
    check(f"(d) {bits}-bit codes in grid range",
          int(r["codes"].max()) < len(grid) and r["codes"].dtype == np.uint8,
          f"max={int(r['codes'].max())} levels={len(grid)}")
    check(f"(d) {bits}-bit dequantize(codes, scales) == w_q exactly",
          np.array_equal(dequantize(r["codes"], r["scales"], bits, 64), r["w_q"]), "")
    check(f"(d) {bits}-bit ragged group shapes",
          r["scales"].shape == (6, 2) and r["codes"].shape == (6, 100)
          and r["w_q"].dtype == np.float32, f"scales={r['scales'].shape}")
    bound = np.abs(scale_per_weight(r["scales"], 100, 64))
    check(f"(d) {bits}-bit |W - w_q| <= |group scale|",
          bool(np.all(np.abs(Wd - r["w_q"]) <= bound + 1e-7)),
          f"max|err|={np.abs(Wd - r['w_q']).max():.4f}")

rt = quantize_rtn(Wd, 1, 64)
ternary_vals = {-1.0, 0.0, 1.0}
s_all = scale_per_weight(rt["scales"], 100, 64)
on_alphabet = np.isin(np.round(rt["w_q"] / np.where(s_all == 0, 1.0, s_all)), [-1.0, 0.0, 1.0])
check("(d) ternary w_q lies in {-s, 0, s} per group",
      bool(np.all(on_alphabet)) and set(np.unique(rt["codes"]).tolist()) <= {0, 1, 2},
      f"codes={sorted(set(np.unique(rt['codes']).tolist()))}")

Hd = np.eye(100)
check("(d) GPTQ with H = I equals RTN codes (no error feedback)",
      np.array_equal(quantize_gptq(Wd, Hd, 3, 64)["codes"], quantize_rtn(Wd, 3, 64)["codes"]), "")

Xd = rng.standard_normal((400, 96))
Xd[:, 5] = 0.0  # dead input: H_55 = 0
Lt = np.tril(rng.standard_normal((300, 300))) + 6.0 * np.eye(300)
check("(d) blocked triangular inverse satisfies L X = I",
      np.abs(Lt @ _tri_inv_lower(Lt, 64) - np.eye(300)).max() < 1e-10, "")

Hdead, _ = hessian_from_inputs(Xd)
Wdead = rng.standard_normal((8, 96)) * 0.05
gd = quantize_gptq(Wdead, Hdead, 2, 32)
check("(d) dead input column quantizes to exactly zero",
      bool(np.all(gd["w_q"][:, 5] == 0.0)) and bool(np.all(gd["codes"][:, 5] == np.argmin(np.abs(get_grid(2))))),
      "")

Xb = rng.standard_normal((512, 96)) @ rng.standard_normal((96, 96)) * 0.1
Hb, _ = hessian_from_inputs(Xb)
Wbk = rng.standard_normal((12, 96)) * 0.05
g_one = quantize_gptq(Wbk, Hb, 2, 24, block_size=96)   # one block, no lazy update
g_blk = quantize_gptq(Wbk, Hb, 2, 24, block_size=16)   # groups straddle blocks
check("(d) GPTQ block-size invariance (lazy batch update is exact)",
      np.array_equal(g_one["codes"], g_blk["codes"]) and
      abs(g_one["err"] - g_blk["err"]) <= 1e-9 * max(g_one["err"], 1e-30),
      f"codes mismatches={int(np.sum(g_one['codes'] != g_blk['codes']))}")

H_a, n_a = hessian_from_inputs(Xb[:200])
H_ab, n_ab = hessian_from_inputs(Xb[200:], H_a, n_a)
H_full, _ = hessian_from_inputs(Xb)
check("(d) streaming Hessian equals one-shot 2 X^T X / M",
      n_ab == 512 and np.allclose(H_ab, H_full, rtol=1e-12, atol=1e-12), "")

try:
    quantize_gptq(Wbk, Hb, 2, 24, act_order=True)
    check("(d) act_order=True is rejected", False, "no error raised")
except NotImplementedError:
    check("(d) act_order=True is rejected", True, "NotImplementedError")

try:
    quantize_rtn(Wd, 9, 64)
    check("(d) bits=9 is rejected (uint8 codes)", False, "no error raised")
except ValueError:
    check("(d) bits=9 is rejected (uint8 codes)", True, "ValueError")

dt = time.time() - T0
print(f"GPTQ TEST OK ({dt:.1f}s)")
assert dt < 90, f"over budget: {dt:.1f}s"
