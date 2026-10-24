"""WP4 acceptance test for the MLX export (src/gsq_rco/gsq/export.py, gsq-rco export).

  (a) packing: every native width (ternary, 2, 3, 4, 5, 6, 8) on random codes. fp32
      scales reproduce s * grid[code] through mx.dequantize to fp32 rounding; bf16
      scales stay within 2^-7 relative per weight; packed words equal mx.quantize's own
      words; the bit accounting matches a hand-made case.
  (c) CLI: gsq-rco export on a tiny random Qwen2, from shards (mixed bits, a ternary
      layer, a group_size key) and with --rtn-bits 4. Non-quantized tensors must be
      byte-identical to the base checkpoint. Also covers the row-blocked RTN, the float32
      scale export and the dense reference loader.
  (b) full model (Qwen2.5-0.5B; the heavy run, started through the shared lock): RTN
      (gptq.quantize_rtn, row blocks) with per-layer bits cycling 2/3/4/8, exported to a
      temp dir. Each stage runs in its own process, so a model's memory is returned when
      its stage exits: (1) export, (2) the exported model, (3) the bf16 model holding the
      same dense weights. Stages hand results over through small files. Every quantized
      module must be a QuantizedLinear with its bits; each linear's output must match the
      dense weights bf16(s) * grid[code]; logits (32 tokens) and NLL (4 x 256 tokens of
      rco.txt) must agree with the dense model. Each stage's peak RSS (kernel rusage) must
      stay under the 1.5 GB budget.

Usage: .venv/bin/python tests/test_export.py     (exit nonzero on any failure)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

import pytest
import sys
import tempfile
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import utils as mlx_lm_utils
from mlx_lm.models import qwen2

from gsq_rco import cli as gsq_cli
from gsq_rco.gsq import export as export_mlx  # rtn_native, load_shards
from gsq_rco.gsq.e2e import WKEY
from gsq_rco.gsq.export import (SafetensorsFile, bits_report, export_mlx_lm, mlx_bits,
                            pack_codes, to_mlx_affine)
from gsq_rco.gsq.gptq import quantize_rtn
from gsq_rco.gsq.quant import get_grid

BASE = os.path.join(ROOT, "models", "Qwen2.5-0.5B")  # bf16 checkpoint
TOKENIZER_JSON = os.path.join(ROOT, "models", "Qwen2.5-0.5B", "tokenizer.json")
GROUP = 64
LAYER_BITS = (2, 3, 4, 8)  # full-model check: layer i gets LAYER_BITS[i % 4]
BOS = 151643  # same windows as gsq.pipeline.build_seqs
# Tolerances, fixed before the first full-model run. bf16 has 8 significand bits (unit
# roundoff 2^-8). A linear's output carries one bf16 output rounding, so its RMS relative
# error should sit near 2^-8 / sqrt(3); 2^-7 leaves a 2x margin.
LINEAR_RMS_TOL = 2.0 ** -7
LOGIT_RMS_TOL = 0.05
ARGMAX_MIN = 0.90
NLL_TOL = 0.02  # nats per token
RSS_BUDGET_MB = 1536  # shared-machine rule: keep each process under ~1.5 GB
STAGES = ("export", "qmodel", "dense")

TINY = dict(model_type="qwen2", hidden_size=128, num_hidden_layers=2, intermediate_size=256,
            num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=512,
            max_position_embeddings=512, rope_theta=10000.0, tie_word_embeddings=True,
            torch_dtype="bfloat16")
TINY_SHAPES = {"q_proj": (128, 128), "k_proj": (64, 128), "v_proj": (64, 128),
               "o_proj": (128, 128), "gate_proj": (256, 128), "up_proj": (256, 128),
               "down_proj": (128, 256)}


def bf16_round(x) -> np.ndarray:
    """Round float32 values to bf16 (as MLX does); returned as float32."""
    return np.array(mx.array(np.asarray(x, dtype=np.float32)).astype(mx.bfloat16)
                    .astype(mx.float32))


def native_weight(codes, scales, bits, group_size=GROUP) -> np.ndarray:
    """Native-format weight with the stored bf16 scales, bf16(s) * grid[code].

    Exact in float32: bf16(s) has 8 significant bits and grid values are integers of at
    most 8 bits, so each product has at most 16 significant bits.
    """
    gid = np.arange(codes.shape[1]) // group_size
    return bf16_round(scales)[:, gid] * get_grid(bits).astype(np.float32)[codes]


def check_packing() -> None:
    rng = np.random.default_rng(0)
    print("(a) packing round trip per native width")
    for bits in (1, 2, 3, 4, 5, 6, 8):
        grid = get_grid(bits).astype(np.float64)
        mb = mlx_bits(bits)
        codes = rng.integers(0, len(grid), size=(8, 256), dtype=np.uint8)
        s = (rng.standard_normal((8, 256 // GROUP)) * 0.1).astype(np.float32)
        ref = s.astype(np.float64)[:, np.arange(256) // GROUP] * grid[codes]
        packed, sc, bi = to_mlx_affine(codes, s, bits, GROUP, "float32")
        w = np.array(mx.dequantize(mx.array(packed), sc, bi, group_size=GROUP, bits=mb))
        tol = 2.0 ** -22 * (1 << mb) * np.abs(s).max()
        err = np.abs(w.astype(np.float64) - ref).max()
        assert err <= tol, (bits, err, tol)

        packed16, sc16, bi16 = to_mlx_affine(codes, s, bits, GROUP, "bfloat16")
        w16 = np.array(mx.dequantize(mx.array(packed16), sc16, bi16, group_size=GROUP,
                                     bits=mb).astype(mx.float32)).astype(np.float64)
        bound = 2.0 ** -7 * (1 + 2.0 ** -6) * np.abs(ref) + tol
        assert np.all(np.abs(w16 - ref) <= bound), bits
        nz = ref != 0
        rel16 = (np.abs(w16 - ref)[nz] / np.abs(ref)[nz]).max()
        print(f"  bits={bits} (MLX {mb}): fp32 max|err| {err:.2e} <= {tol:.2e}; "
              f"bf16 max rel err {rel16:.2e}")

    rng = np.random.default_rng(1)
    for bits in (2, 3, 4, 5, 6, 8):
        codes = rng.integers(0, 1 << bits, size=(4, 256), dtype=np.uint8)
        codes[:, ::GROUP] = 0
        codes[:, 1::GROUP] = (1 << bits) - 1
        words, msc, mbi = mx.quantize(mx.array(codes.astype(np.float32)),
                                      group_size=GROUP, bits=bits)
        gid = np.arange(256) // GROUP
        msc64 = np.array(msc).astype(np.float64)[:, gid]
        mbi64 = np.array(mbi).astype(np.float64)[:, gid]
        q_mlx = np.rint((codes - mbi64) / msc64).astype(np.uint8)  # MLX's own codes
        assert q_mlx.max() < (1 << bits)
        assert np.array_equal(pack_codes(q_mlx, bits), np.array(words)), bits
    print("  packed words equal mx.quantize's words for bits 2, 3, 4, 5, 6, 8")

    raised = False
    try:
        to_mlx_affine(np.full((2, 64), 4, np.uint8), np.ones((2, 1), np.float32), 2)
    except ValueError:
        raised = True
    assert raised, "out-of-range codes must be rejected"

    hand = {"a": dict(codes=np.zeros((4, 128), np.uint8),
                      scales=np.ones((4, 2), np.float16), bits=2, group_size=64),
            "b": dict(codes=np.zeros((2, 64), np.uint8),
                      scales=np.ones((2, 1), np.float16), bits=1, group_size=64)}
    rep = bits_report(hand, 640)
    code_bpp = (512 * 2 + 128 * np.log2(3)) / 640
    assert np.isclose(rep["code_bpp"], code_bpp)
    assert np.isclose(rep["native_bpp"], code_bpp + 16 * 10 / 640)
    assert np.isclose(rep["mlx_bpp"], (512 * 2 + 128 * 2 + 2 * 16 * 10) / 640)
    assert np.isclose(rep["shard_bpp"], (8 * 640 + 16 * 10) / 640)
    print(f"  bits_report on a hand-made case: code {rep['code_bpp']:.4f}, "
          f"native {rep['native_bpp']:.4f}, mlx {rep['mlx_bpp']:.4f} (arithmetic checked)")


def _tiny_base(d: str) -> None:
    os.makedirs(d, exist_ok=True)
    model = qwen2.Model(qwen2.ModelArgs.from_dict(TINY))
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(d, "model.safetensors"),
                        {k: v.astype(mx.bfloat16) for k, v in flat.items()})
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(TINY, f)


def _write_shards(d: str, layer_bits) -> dict:
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(2)
    expected = {}
    for layer, bits in enumerate(layer_bits):
        shard = {"bits": np.array(bits), "grid": get_grid(bits).astype(np.float32)}
        if layer == 1:  # tolerated optional key
            shard["group_size"] = np.array(GROUP)
        for short, (o, i) in TINY_SHAPES.items():
            codes = rng.integers(0, len(get_grid(bits)), size=(o, i), dtype=np.uint8)
            scales = (rng.standard_normal((o, i // GROUP)) * 0.05).astype(np.float16)
            shard[f"{short}.codes"] = codes
            shard[f"{short}.scales"] = scales
            sub = WKEY[short][: -len(".weight")]
            expected[f"model.layers.{layer}.{sub}"] = (codes, scales, bits)
        np.savez(os.path.join(d, f"layer_{layer:02d}.npz"), **shard)
    return expected


def _check_export(out: str, base: str, expected: dict) -> None:
    model, cfg = mlx_lm_utils.load_model(pathlib.Path(out))
    leaves = dict(tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module))
    for path, (codes, scales, bits) in expected.items():
        m = leaves[path]
        assert isinstance(m, nn.QuantizedLinear), path
        assert m.bits == mlx_bits(bits) and m.group_size == GROUP, (path, m.bits)
        assert cfg["quantization"][path] == {"group_size": GROUP, "bits": mlx_bits(bits)}
        want = native_weight(codes, scales, bits)
        got = np.array(mx.dequantize(m.weight, m.scales, m.biases, group_size=GROUP,
                                     bits=m.bits).astype(mx.float32))
        assert np.all(np.abs(got - want) <= 2.0 ** -7 * np.abs(want)), path
    b_st = SafetensorsFile(os.path.join(base, "model.safetensors"))
    o_st = SafetensorsFile(os.path.join(out, "model.safetensors"))
    quantized_names = {p + ".weight" for p in expected}
    n_copied = 0
    for name in b_st.names():
        if name not in quantized_names:
            assert b_st.read_bytes(name) == o_st.read_bytes(name), name
            n_copied += 1
    print(f"  {len(expected)} linears loaded as QuantizedLinear with the expected bits and "
          f"dequantized values; {n_copied} other tensors byte-identical to base")


def check_cli(tmp: str) -> None:
    print("(c) CLI on a tiny random Qwen2")
    W = np.random.default_rng(4).standard_normal((50, 128)).astype(np.float32)
    one = quantize_rtn(W, 3, GROUP)
    blk_codes, blk_scales = export_mlx.rtn_native(W, 3, GROUP, rows=7)  # ragged last block
    assert np.array_equal(blk_codes, one["codes"]) and np.array_equal(blk_scales, one["scales"])
    print("  row-blocked RTN equals one-shot quantize_rtn (50 rows in blocks of 7)")
    base = os.path.join(tmp, "tiny_base")
    shards = os.path.join(tmp, "shards")
    _tiny_base(base)
    expected = _write_shards(shards, (1, 4))  # layer 0 ternary (exported as 2), layer 1 4-bit
    out = os.path.join(tmp, "tiny_shards_mlx")
    assert gsq_cli.main(["export", "--base", base, "--shards", shards, "--out", out]) == 0
    print("  shards path:")
    _check_export(out, base, expected)

    st = SafetensorsFile(os.path.join(base, "model.safetensors"))
    expected_rtn = {}
    for layer in range(TINY["num_hidden_layers"]):
        for short in WKEY:
            sub = WKEY[short][: -len(".weight")]
            q = quantize_rtn(st.tensor_f32(f"model.layers.{layer}.{sub}.weight"), 4, GROUP)
            expected_rtn[f"model.layers.{layer}.{sub}"] = (q["codes"], q["scales"], 4)
    out_rtn = os.path.join(tmp, "tiny_rtn4_mlx")
    assert gsq_cli.main(["export", "--base", base, "--out", out_rtn, "--rtn-bits", "4"]) == 0
    print("  --rtn-bits 4 path (row-blocked RTN, compared with one-shot quantize_rtn):")
    _check_export(out_rtn, base, expected_rtn)

    model, _ = mlx_lm_utils.load_model(pathlib.Path(base))
    load_rtn_weights(model, st, TINY["num_hidden_layers"])
    params = dict(tree_flatten(model.parameters()))
    for path, q in iter_rtn(st, TINY["num_hidden_layers"]):
        got = np.array(params[path + ".weight"].astype(mx.float32))
        assert np.array_equal(got, bf16_round(native_weight(q["codes"], q["scales"],
                                                            q["bits"]))), path
    print("  load_rtn_weights (dense reference) equals bf16 of the native weights")

    out32 = os.path.join(tmp, "tiny_shards_fp32_mlx")
    export_mlx_lm(out32, export_mlx.load_shards(shards), base, scale_dtype="float32")
    model32, _ = mlx_lm_utils.load_model(pathlib.Path(out32))
    leaves32 = dict(tree_flatten(model32.leaf_modules(), is_leaf=nn.Module.is_module))
    for path, (codes, scales, bits) in expected.items():
        m = leaves32[path]
        assert m.scales.dtype == mx.float32, path
        gid = np.arange(codes.shape[1]) // GROUP
        want = scales.astype(np.float64)[:, gid] * get_grid(bits).astype(np.float64)[codes]
        got = np.array(mx.dequantize(m.weight, m.scales, m.biases, group_size=GROUP,
                                     bits=m.bits)).astype(np.float64)
        assert np.abs(got - want).max() <= 2.0 ** -22 * (1 << m.bits) * np.abs(scales).max(), path
    print("  scale_dtype=float32 export: scales stay float32 and dequantize to s * grid")


def eval_data():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    with open(os.path.join(ROOT, "rco.txt")) as f:
        ids = tok.encode(f.read()).ids
    assert len(ids) >= 4 * 255, "rco.txt too short for 4 x 256 windows"
    windows = [np.array([BOS] + ids[i * 255:(i + 1) * 255], dtype=np.int64) for i in range(4)]
    short = np.array([BOS] + ids[:31], dtype=np.int64)
    return short, windows


def run_eval(model, short, windows):
    """Logits for the short text, and per-token NLL for each window ([n_windows, 255])."""
    logits = np.array(model(mx.array(short[None])).astype(mx.float32))[0]
    rows = []
    for w in windows:
        x = mx.array(w[None])
        lg = model(x[:, :-1])
        tgt = x[:, 1:]
        parts = []
        for c0 in range(0, lg.shape[1], 32):  # chunked fp32 log-softmax keeps memory low
            c = lg[:, c0:c0 + 32].astype(mx.float32)
            picked = mx.take_along_axis(c, tgt[:, c0:c0 + 32, None], axis=-1)[..., 0]
            parts.append(mx.logsumexp(c, axis=-1) - picked)
        row = mx.concatenate(parts, axis=1)
        mx.eval(row)
        rows.append(np.array(row)[0].astype(np.float64))
    return logits, np.stack(rows)


def iter_rtn(st: SafetensorsFile, n_layers: int):
    """Yield (module path, RTN quantization with bits LAYER_BITS[layer % 4]) per base linear."""
    for layer in range(n_layers):
        bits = LAYER_BITS[layer % len(LAYER_BITS)]
        for short in WKEY:
            path = f"model.layers.{layer}." + WKEY[short][: -len(".weight")]
            codes, scales = export_mlx.rtn_native(st.tensor_f32(path + ".weight"), bits, GROUP)
            yield path, dict(codes=codes, scales=scales, bits=bits, group_size=GROUP)


def load_rtn_weights(model, st: SafetensorsFile, n_layers: int) -> None:
    """Load each base linear as the dense bf16 weight bf16(s) * grid[code].

    Each weight is materialized before the next one is built. A lazy cast keeps its fp32
    input alive until one final eval; that held 1.4 GB extra in the first full-model run.
    """
    for path, q in iter_rtn(st, n_layers):
        w = mx.array(native_weight(q["codes"], q["scales"], q["bits"]))
        w = w.astype(mx.bfloat16)
        mx.eval(w)
        model.load_weights([(path + ".weight", w)], strict=False)


def _n_layers(base: str) -> int:
    with open(os.path.join(base, "config.json")) as f:
        return json.load(f)["num_hidden_layers"]


def stage_export(base: str, tmp: str) -> None:
    """Stage 1: quantize (RTN, LAYER_BITS) and export. No model is loaded here."""
    st = SafetensorsFile(os.path.join(base, "model.safetensors"))
    quant = dict(iter_rtn(st, _n_layers(base)))
    rep = bits_report(quant, sum(q["codes"].size for q in quant.values()))
    layout = {p: [list(q["codes"].shape), q["bits"]] for p, q in quant.items()}
    out_dir = os.path.join(tmp, "qwen_mlx")
    export_mlx_lm(out_dir, quant, base)
    with open(os.path.join(tmp, "export.json"), "w") as f:
        json.dump({"layout": layout, "rep": rep,
                   "size": os.path.getsize(os.path.join(out_dir, "model.safetensors"))}, f)


def stage_qmodel(base: str, tmp: str) -> None:
    """Stage 2: the exported model. Structure, per-linear outputs, logits and NLL."""
    with open(os.path.join(tmp, "export.json")) as f:
        layout = {p: (tuple(sh), bits) for p, (sh, bits) in json.load(f)["layout"].items()}
    out_dir = os.path.join(tmp, "qwen_mlx")
    with open(os.path.join(out_dir, "config.json")) as f:
        cfg = json.load(f)
    for path, (_, bits) in layout.items():
        assert cfg["quantization"][path] == {"group_size": GROUP, "bits": mlx_bits(bits)}, path
    short, windows = eval_data()

    qmodel, _ = mlx_lm_utils.load_model(pathlib.Path(out_dir))
    leaves = dict(tree_flatten(qmodel.leaf_modules(), is_leaf=nn.Module.is_module))
    assert isinstance(leaves["model.embed_tokens"], nn.Embedding)
    for path, ((out_f, in_f), bits) in layout.items():
        m = leaves[path]
        mb = mlx_bits(bits)
        assert isinstance(m, nn.QuantizedLinear), path
        assert m.bits == mb and m.group_size == GROUP, (path, m.bits)
        assert m.weight.dtype == mx.uint32 and m.weight.shape == (out_f, in_f * mb // 32), path
    print(f"  all {len(layout)} linears are QuantizedLinear with the configured bits; "
          f"embeddings unquantized")

    st = SafetensorsFile(os.path.join(base, "model.safetensors"))
    rng = np.random.default_rng(3)
    worst = 0.0
    for path, q in iter_rtn(st, _n_layers(base)):  # one linear's codes at a time
        m = leaves[path]
        in_f = q["codes"].shape[1]
        x = mx.array(rng.standard_normal((1, 16, in_f)).astype(np.float32)).astype(mx.bfloat16)
        y = np.array(m(x).astype(mx.float32))[0].astype(np.float64)
        ref = np.array(x.astype(mx.float32))[0].astype(np.float64) @ native_weight(
            q["codes"], q["scales"], q["bits"]).astype(np.float64).T
        if getattr(m, "bias", None) is not None:
            ref += np.array(m.bias.astype(mx.float32)).astype(np.float64)
        worst = max(worst, np.linalg.norm(y - ref) / np.linalg.norm(ref))
    print(f"  per-linear output vs dense weights: worst RMS rel error {worst:.2e} "
          f"(tol {LINEAR_RMS_TOL:.2e})")

    q_logits, q_nll = run_eval(qmodel, short, windows)
    np.savez(os.path.join(tmp, "qmodel.npz"), logits=q_logits, nll=q_nll)
    with open(os.path.join(tmp, "qmodel.json"), "w") as f:
        json.dump({"worst_linear": float(worst)}, f)


def stage_dense(base: str, tmp: str) -> None:
    """Stage 3: the bf16 model holding the same dense weights, and the FP model before it."""
    short, windows = eval_data()
    st = SafetensorsFile(os.path.join(base, "model.safetensors"))
    base_model, _ = mlx_lm_utils.load_model(pathlib.Path(base))
    fp_logits, fp_nll = run_eval(base_model, short, windows)
    load_rtn_weights(base_model, st, _n_layers(base))
    r_logits, r_nll = run_eval(base_model, short, windows)
    np.savez(os.path.join(tmp, "dense.npz"), fp_logits=fp_logits, fp_nll=fp_nll,
             r_logits=r_logits, r_nll=r_nll)


def _subprocess_env() -> dict:
    """Stage processes import gsq_rco from this checkout's src."""
    src = os.path.join(ROOT, "src")
    return dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))


def _run_stage(stage: str, tmp: str, base: str) -> float:
    """Run one stage in its own process. Returns that process's peak RSS in MB (kernel)."""
    sys.stdout.flush()
    proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--stage", stage,
                             "--tmp", tmp, "--base", base], env=_subprocess_env())
    _, status, ru = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)
    if proc.returncode:
        raise RuntimeError(f"stage {stage!r} exited with code {proc.returncode}")
    return ru.ru_maxrss / 2**20 if sys.platform == "darwin" else ru.ru_maxrss / 1024


def check_full_model(tmp: str, base: str = BASE) -> None:
    print("(b) full model: Qwen2.5-0.5B, RTN with per-layer bits 2/3/4/8 cycling")
    t0 = time.time()
    rss = {stage: _run_stage(stage, tmp, base) for stage in STAGES}
    with open(os.path.join(tmp, "export.json")) as f:
        ex = json.load(f)
    with open(os.path.join(tmp, "qmodel.json")) as f:
        qm = json.load(f)
    qz = np.load(os.path.join(tmp, "qmodel.npz"))
    dz = np.load(os.path.join(tmp, "dense.npz"))
    rep = ex["rep"]
    print(f"  bpp: code {rep['code_bpp']:.3f} | native (fp16 scales) {rep['native_bpp']:.3f} | "
          f"MLX export (bf16 scale+bias) {rep['mlx_bpp']:.3f} | current shards "
          f"{rep['shard_bpp']:.3f}")
    print(f"  export: model.safetensors {ex['size'] / 1e6:.1f} MB "
          f"(quantized linears {rep['mlx_bytes'] / 1e6:.1f} MB)")

    def rel(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    q_logits, q_nll = qz["logits"], qz["nll"]
    fp_logits, fp_nll = dz["fp_logits"], dz["fp_nll"]
    r_logits, r_nll = dz["r_logits"], dz["r_nll"]
    assert qm["worst_linear"] < LINEAR_RMS_TOL, qm["worst_linear"]
    logit_rel = rel(q_logits, r_logits)
    agree = float(np.mean(q_logits.argmax(-1) == r_logits.argmax(-1)))
    nll_diff = abs(q_nll.mean() - r_nll.mean())
    assert np.isfinite(q_logits).all() and np.isfinite(q_nll).all()
    assert logit_rel < LOGIT_RMS_TOL, logit_rel
    assert agree >= ARGMAX_MIN, agree
    assert nll_diff < NLL_TOL, nll_diff
    assert rel(q_logits, fp_logits) > 0.1, "quantized logits should differ from FP"
    print(f"  logits, 32 tokens: RMS rel diff quantized vs dense-same-weights {logit_rel:.2e} "
          f"(tol {LOGIT_RMS_TOL}); argmax agreement {agree:.3f} (min {ARGMAX_MIN})")
    print(f"  NLL per token (4 x 255 targets): FP {fp_nll.mean():.4f} (perplexity "
          f"{np.exp(fp_nll.mean()):.2f}), dense-same-weights {r_nll.mean():.4f}, quantized "
          f"{q_nll.mean():.4f}; |quantized - dense| {nll_diff:.4f} (tol {NLL_TOL})")
    print(f"  non-vacuity: quantized vs FP logits RMS rel diff "
          f"{rel(q_logits, fp_logits):.3f} (> 0.1)")
    print(f"  peak RSS per stage (kernel rusage): export {rss['export']:.0f} MB | "
          f"exported model {rss['qmodel']:.0f} MB | dense reference {rss['dense']:.0f} MB "
          f"(budget {RSS_BUDGET_MB} MB)")
    print(f"  time {time.time() - t0:.0f}s")
    for stage, peak in rss.items():
        assert peak < RSS_BUDGET_MB, f"stage {stage!r} peak RSS {peak:.0f} MB over budget"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WP4 export acceptance test")
    ap.add_argument("--stage", choices=STAGES, help=argparse.SUPPRESS)  # internal
    ap.add_argument("--tmp", help=argparse.SUPPRESS)
    ap.add_argument("--base", default=BASE, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.stage:  # one full-model stage, in its own process
        {"export": stage_export, "qmodel": stage_qmodel, "dense": stage_dense}[args.stage](
            args.base, args.tmp)
        return 0
    t0 = time.time()
    check_packing()
    print("  PASS")
    with tempfile.TemporaryDirectory(prefix="wp4_test_") as tmp:
        check_cli(tmp)
        print("  PASS")
        check_full_model(tmp)
        print("  PASS")
    print(f"ALL PASS in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_export_packing():
    check_packing()


def test_export_cli(tmp_path):
    check_cli(str(tmp_path))


@pytest.mark.slow
def test_export_full_model(tmp_path):
    check_full_model(str(tmp_path))


def test_load_shards_reads_gsq_pipeline_format(tmp_path):
    """Shards written by gsq.pipeline carry bits per linear and the group size in meta; the export
    reads each linear at its own bit-width (the WP5 request)."""
    from gsq_rco.gsq.pipeline import save_shard
    from gsq_rco.gsq.export import load_shards
    bits = {"q_proj": 2, "k_proj": 3, "v_proj": 4, "o_proj": 2, "gate_proj": 3, "up_proj": 4,
            "down_proj": 2}
    rng = np.random.default_rng(5)
    lins = {}
    for short, (o, i) in TINY_SHAPES.items():
        b = bits[short]
        codes = rng.integers(0, len(get_grid(b)), size=(o, i), dtype=np.uint8)
        scales = (rng.standard_normal((o, i // GROUP)) * 0.05).astype(np.float16)
        lins[short] = {"codes": codes, "scales": scales, "bits": b}
    fp = {"model": "tiny", "group_size": GROUP}
    save_shard(str(tmp_path / "layer_00.npz"), 0, lins, bits, fp, {"note": "test"})
    loaded = load_shards(str(tmp_path))
    assert len(loaded) == len(TINY_SHAPES)
    for short in TINY_SHAPES:
        path = f"model.layers.0.{WKEY[short][: -len('.weight')]}"
        q = loaded[path]
        assert q["bits"] == bits[short], (path, q["bits"])
        assert q["group_size"] == GROUP
        assert q["scales"].dtype == np.float32
        assert np.array_equal(q["codes"], lins[short]["codes"])
        assert np.array_equal(q["scales"], lins[short]["scales"].astype(np.float32))
