"""Unit tests for the shared data, model and evaluation layer (WP3).

Usage: .venv/bin/python scripts/test_common.py   (exit 1 if any check fails)

Synthetic checks use tiny arrays and a randomly initialised 3-block Qwen2 (the
mlx-lm class the real model uses). The real-data checks need WikiText-2 (downloaded
into <repo>/data/ on first use) and the 1 GB FP checkpoint, loaded once.
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.parquet as pq
from mlx.utils import tree_map
from mlx_lm.models.qwen2 import Model, ModelArgs

from gsq_rco_common import data, evaluate
from gsq_rco_common import model as qm

RESULTS: list[tuple[str, bool]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def tiny_model(seed: int = 0, dtype=mx.float32):
    mx.random.seed(seed)
    args = ModelArgs(model_type="qwen2", hidden_size=32, num_hidden_layers=3,
                     intermediate_size=48, num_attention_heads=4,
                     num_key_value_heads=2, rms_norm_eps=1e-6, vocab_size=97,
                     rope_theta=10000.0, tie_word_embeddings=True)
    model = Model(args)
    if dtype != mx.float32:
        model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    return model


def test_grid_and_sampling():
    tokens = np.arange(1000, dtype=np.int32)
    grid = data._grid(tokens, 64)
    check("grid: 15 non-overlapping windows, tail dropped",
          grid.shape == (15, 64) and grid.dtype == np.int32
          and np.array_equal(grid.reshape(-1), tokens[:960]),
          f"shape={grid.shape}")
    a = data._sample_rows(grid, 8, seed=0)
    starts = a[:, 0] // 64
    check("sampled rows are distinct grid windows",
          np.array_equal(a, grid[starts]) and len(set(starts.tolist())) == 8)
    check("sampling is deterministic per seed",
          np.array_equal(a, data._sample_rows(grid, 8, seed=0)))
    check("sampling depends on the seed",
          not np.array_equal(a, data._sample_rows(grid, 8, seed=1)))
    try:
        data._sample_rows(grid, 16, seed=0)
        rejected = False
    except ValueError:
        rejected = True
    check("sampling rejects more windows than exist", rejected)
    try:
        data.eval_windows(2048, split="dev")
        rejected = False
    except ValueError:
        rejected = True
    check("unknown split rejected before any download", rejected)


def test_real_data():
    rows = {s: pq.read_table(data._split_parquet(s), columns=["text"]).num_rows
            for s in data.SPLITS}
    check("WikiText-2 raw v1 row counts",
          rows == {"train": 36718, "validation": 3760, "test": 4358}, str(rows))
    n_test = len(data.tokenize_split("test"))
    ev = data.eval_windows(2048, "test")
    check("eval_windows: shape/dtype, remainder dropped",
          ev.shape == (n_test // 2048, 2048) and ev.dtype == np.int32,
          f"shape={ev.shape} test_tokens={n_test}")
    check("eval_windows: max_windows gives a prefix",
          np.array_equal(data.eval_windows(2048, "test", max_windows=3), ev[:3]))
    check("token ids inside the Qwen2.5 vocabulary",
          int(ev.min()) >= 0 and int(ev.max()) < 151936)
    c0 = data.calibration_windows(8, 2048, seed=0)
    check("calibration_windows: shape/dtype",
          c0.shape == (8, 2048) and c0.dtype == np.int32, f"shape={c0.shape}")
    check("calibration_windows: deterministic for a seed",
          np.array_equal(c0, data.calibration_windows(8, 2048, seed=0)))
    check("calibration_windows: seed changes the draw",
          not np.array_equal(c0, data.calibration_windows(8, 2048, seed=1)))
    check("calibration_windows: rows are distinct",
          len({r.tobytes() for r in c0}) == 8)


def test_nll_and_perplexity():
    model = tiny_model(0)
    rng = np.random.default_rng(0)
    h_np = rng.standard_normal((2, 10, 32)).astype(np.float32)
    tg = rng.integers(0, 97, size=(2, 10))
    s_chunked, n = evaluate.nll_from_hidden(model, mx.array(h_np), tg, chunk=3)
    s_whole, _ = evaluate.nll_from_hidden(model, mx.array(h_np), tg, chunk=1000)
    E = np.asarray(model.model.embed_tokens.weight, dtype=np.float64)
    logits = h_np.reshape(-1, 32).astype(np.float64) @ E.T
    m = logits.max(-1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(-1))
    ref = float((lse - logits[np.arange(20), tg.reshape(-1)]).sum())
    check("chunked NLL == unchunked NLL",
          abs(float(s_chunked) - float(s_whole)) <= 1e-5 * abs(ref))
    check("chunked NLL == float64 numpy reference",
          abs(float(s_chunked) - ref) <= 1e-4 * abs(ref),
          f"{float(s_chunked):.6f} vs {ref:.6f}")
    check("NLL counts B*T positions", n == 20)

    W = rng.integers(0, 97, size=(4, 17)).astype(np.int32)
    res = evaluate.perplexity(model, W, chunk=5)
    x = mx.array(W)
    ce = nn.losses.cross_entropy(model(x[:, :-1]).astype(mx.float32),
                                 x[:, 1:], reduction="mean")
    mx.eval(ce)
    check("perplexity == mlx-lm full-logit cross entropy",
          math.isclose(res["ppl"], math.exp(float(ce)), rel_tol=1e-4),
          f"{res['ppl']:.4f} vs {math.exp(float(ce)):.4f}")
    check("perplexity counts S-1 predictions per window",
          res["n_tokens"] == 4 * 16 and res["n_windows"] == 4
          and res["seq_len"] == 17, f"n_tokens={res['n_tokens']}")


def _manual_kl_mean(model, hq, hr):
    """KL(p_ref || p_q) on full logits with plain autodiff: the reference."""
    head = model.model.embed_tokens.as_linear

    def log_softmax(z):
        z = z.astype(mx.float32)
        return z - mx.logsumexp(z, axis=-1, keepdims=True)

    lq = log_softmax(head(hq))
    lr = log_softmax(head(hr))
    return mx.mean(mx.sum(mx.exp(lr) * (lr - lq), axis=-1))


def test_kl():
    model = tiny_model(2)
    rng = np.random.default_rng(2)
    hq_np = rng.standard_normal((2, 6, 32)).astype(np.float32)
    hr_np = rng.standard_normal((2, 6, 32)).astype(np.float32)
    hq, hr = mx.array(hq_np), mx.array(hr_np)

    self_kl = float(evaluate.kl_from_hidden(model, hr, hr, chunk=4))
    check("KL(p || p) == 0 for identical inputs", self_kl == 0.0, f"{self_kl!r}")

    kl = float(evaluate.kl_from_hidden(model, hq, hr, chunk=4))
    E = np.asarray(model.model.embed_tokens.weight, dtype=np.float64)

    def logp(h):
        z = h.reshape(-1, 32).astype(np.float64) @ E.T
        z -= z.max(-1, keepdims=True)
        return z - np.log(np.exp(z).sum(-1, keepdims=True))

    lr_, lq_ = logp(hr_np), logp(hq_np)
    ref = float(np.mean(np.sum(np.exp(lr_) * (lr_ - lq_), axis=-1)))
    check("KL matches float64 numpy reference",
          abs(kl - ref) <= 1e-4 * abs(ref), f"{kl:.6f} vs {ref:.6f}")
    kls = [float(evaluate.kl_from_hidden(
        model, mx.array(rng.standard_normal((2, 6, 32)).astype(np.float32)),
        hr, chunk=4)) for _ in range(5)]
    check("KL >= 0 across random pairs (1e-6 float tolerance)",
          min(kls) >= -1e-6, f"min={min(kls):.3e}")
    kl_whole = float(evaluate.kl_from_hidden(model, hq, hr, chunk=1000))
    check("KL is chunk-invariant", abs(kl - kl_whole) <= 1e-6 * max(1.0, abs(kl)))

    g_ckpt = mx.grad(lambda h: evaluate.kl_from_hidden(model, h, hr, chunk=4))(hq)
    g_ref = mx.grad(lambda h: _manual_kl_mean(model, h, hr))(hq)
    g_zero = mx.grad(lambda h: evaluate.kl_from_hidden(model, h, hr, chunk=4))(hr)
    mx.eval(g_ckpt, g_ref, g_zero)
    gscale = float(mx.max(mx.abs(g_ref)))
    gerr = float(mx.max(mx.abs(g_ckpt - g_ref)))
    check("KL gradient w.r.t. h_q matches full-logit autodiff",
          gerr <= 1e-5 * max(1.0, gscale), f"max|dg|={gerr:.2e} max|g|={gscale:.2e}")
    check("KL gradient vanishes at h_q == h_ref",
          float(mx.max(mx.abs(g_zero))) <= 1e-6,
          f"max|g|={float(mx.max(mx.abs(g_zero))):.2e}")

    v_cg, g_cg = evaluate.kl_value_and_grad_hidden(model, hq, hr, chunk=4)
    v_cg1, g_cg1 = evaluate.kl_value_and_grad_hidden(model, hq, hr, chunk=1000)
    mx.eval(v_cg, g_cg, v_cg1, g_cg1)
    check("chunk-wise KL value == kl_from_hidden value",
          abs(float(v_cg) - kl) <= 1e-6 * max(1.0, abs(kl)),
          f"{float(v_cg):.8f} vs {kl:.8f}")
    check("chunk-wise KL gradient matches full-logit autodiff",
          float(mx.max(mx.abs(g_cg - g_ref))) <= 1e-5 * max(1.0, gscale),
          f"max|dg|={float(mx.max(mx.abs(g_cg - g_ref))):.2e}")
    check("chunk-wise KL gradient is chunk-invariant",
          float(mx.max(mx.abs(g_cg - g_cg1))) <= 1e-6 * max(1.0, gscale))


def test_swapped_weights():
    model = tiny_model(3, dtype=mx.bfloat16)
    orig = qm.get_linear_weight(model, 1, "o_proj")
    orig_np = np.asarray(orig.astype(mx.float32))
    W = mx.full(orig.shape, 0.5, dtype=mx.float32)
    with qm.swapped_weights(model, {(1, "o_proj"): W}):
        inside = qm.get_linear_weight(model, 1, "o_proj")
        check("swapped weight is installed, cast to model dtype",
              inside.dtype == mx.bfloat16
              and float(mx.max(mx.abs(inside.astype(mx.float32) - 0.5))) == 0.0,
              f"dtype={inside.dtype}")
    after = qm.get_linear_weight(model, 1, "o_proj")
    check("swapped_weights restores originals on exit",
          np.array_equal(np.asarray(after.astype(mx.float32)), orig_np))

    q_orig = np.asarray(qm.get_linear_weight(model, 2, "q_proj").astype(mx.float32))
    try:
        with qm.swapped_weights(model, {(2, "q_proj"): W}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    check("swapped_weights restores originals when the body raises",
          np.array_equal(
              np.asarray(qm.get_linear_weight(model, 2, "q_proj").astype(mx.float32)),
              q_orig))

    try:
        with qm.swapped_weights(model, {(0, "q_proj"): mx.zeros((3, 3))}):
            pass
        rejected = False
    except ValueError:
        rejected = True
    check("swapped_weights rejects a wrong shape", rejected)


def test_real_model():
    groups = qm.linear_groups()
    check("linear_groups: 168 unique (layer, name) pairs in order",
          len(groups) == 168 and len(set(groups)) == 168
          and groups[0] == (0, "q_proj") and groups[-1] == (23, "down_proj"))
    check("linear_path naming",
          qm.linear_path(3, "q_proj") == "model.layers.3.self_attn.q_proj"
          and qm.linear_path(3, "down_proj") == "model.layers.3.mlp.down_proj")

    model = qm.load_fp_model()
    expected = {"q_proj": (896, 896), "k_proj": (128, 896), "v_proj": (128, 896),
                "o_proj": (896, 896), "gate_proj": (4864, 896),
                "up_proj": (4864, 896), "down_proj": (896, 4864)}
    shapes = {n: tuple(qm.get_linear_weight(model, 0, n).shape) for n in qm.LINEARS}
    check("real checkpoint: linear shapes match config", shapes == expected)
    check("real checkpoint: all 168 linears resolve in bf16",
          all(qm.get_linear_weight(model, i, n).dtype == mx.bfloat16
              for i, n in groups))


TESTS = [test_grid_and_sampling, test_real_data, test_nll_and_perplexity,
         test_kl, test_swapped_weights, test_real_model]


def main():
    t0 = time.time()
    for fn in TESTS:
        try:
            fn()
        except Exception as e:  # keep going so every failure is visible
            check(f"{fn.__name__} raised", False, f"{type(e).__name__}: {e}")
    n_fail = sum(not ok for _, ok in RESULTS)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed "
          f"({time.time() - t0:.1f}s)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
