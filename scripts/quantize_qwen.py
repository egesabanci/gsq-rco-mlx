"""GSQ quantize + eval for Qwen2.5-0.5B (OWNED by GSQ side).

Modes:
  synthetic (default, Phase-1 stub): shapes from config.json + synthetic
      weights only. Never touches model.safetensors.
      Usage: .venv/bin/python scripts/quantize_qwen.py [--bits 2]
  real (Phase-2b end-to-end): streaming layer-by-layer quantize + eval on
      the REAL checkpoint under a ~70MB ceiling. Weights are read
      per-tensor via pread (never fully materialized); batch-1 forwards;
      per-layer shard npz under results/.eval_cache (git-ignored);
      final export results/qwen2.5-0.5b-gsq3bit.npz (git-ignored) and
      results/gsq_eval.json.
      Usage: .venv/bin/python scripts/quantize_qwen.py --mode real
             [--layers 0-23] [--steps 20] [--resamples 2] [--resume]

NEVER touches src/rco_mlx/* or scripts/allocate_qwen.py.
"""
from __future__ import annotations
import argparse
import io
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
CONFIG = os.path.join(MODEL_DIR, "config.json")
SAFETENSORS = os.path.join(MODEL_DIR, "model.safetensors")
ALLOC = os.path.join(ROOT, "results", "rco_allocation.json")
EVAL_CACHE = os.path.join(ROOT, "results", ".eval_cache")
FINAL_NPZ = os.path.join(ROOT, "results", "qwen2.5-0.5b-gsq3bit.npz")
EVAL_JSON = os.path.join(ROOT, "results", "gsq_eval.json")
PARAMS_PER_LAYER = 14909440


# ------------------------------------------------------------- synthetic ---

def run_synthetic(args) -> None:
    from gsq_mlx.gptq import load_qwen_shapes, init_layer, synthetic_weights
    from gsq_mlx.quant import get_grid
    shapes = load_qwen_shapes(args.config)
    grid = get_grid(args.bits)
    print(f"Qwen2.5-0.5B per-block linears (from config.json only): {shapes}")
    print(f"bits={args.bits} grid={list(map(float, grid))} group_size={args.group_size}")
    total = 0
    for i, (name, (out_f, in_f)) in enumerate(shapes.items()):
        W = synthetic_weights((out_f, in_f), seed=1000 + i)  # stand-in
        init = init_layer(args.bits, (out_f, in_f), args.group_size,
                          seed=i, w=W)
        mse = float(((init["w_q"] - W) ** 2).mean())
        sc = init["scales"]
        n = out_f * in_f
        total += n
        print(f"  {name:10s} [{out_f}x{in_f}] params={n:7d} "
              f"groups={sc.size:5d} scale_mean={sc.mean():.5f} rtn_mse={mse:.6f}")
    print(f"per-block params={total} (x24 blocks, embeddings excluded in Phase 2)")
    print("Phase 1 OK: RTN init + GSQ logits built from synthetic weights only.")
    print("Phase 2: load real weights, calibration tokens, staged block loop.")


# ------------------------------------------------------------------ real ---

def parse_layers(s: str):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def build_seqs(n_cal=8, n_eval=4, seq_len=256):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
    bos = 151643
    cal_ids = tok.encode(open(os.path.join(ROOT, "gsq.txt")).read()).ids
    eva_ids = tok.encode(open(os.path.join(ROOT, "rco.txt")).read()).ids
    cal = [np.array([bos] + cal_ids[i * (seq_len - 1):(i + 1) * (seq_len - 1)],
                    dtype=np.int64) for i in range(n_cal)]
    eva = [np.array([bos] + eva_ids[i * (seq_len - 1):(i + 1) * (seq_len - 1)],
                    dtype=np.int64) for i in range(n_eval)]
    assert all(len(s) == seq_len for s in cal + eva), "corpus too short?"
    return cal, eva


def run_real(args) -> None:
    import mlx.core as mx
    from gsq_mlx.e2e import (LazyWeights, Qwen2Stream, train_linear_mx,
                             dequant_linear, seq_nll, rmsnorm,
                             LIN_ORDER, WKEY, BKEY)
    from gsq_mlx.quant import get_grid
    t_all = time.time()
    os.makedirs(EVAL_CACHE, exist_ok=True)
    mx.random.seed(args.seed)

    alloc = json.load(open(ALLOC))
    per_layer_bits = {int(k): int(v) for k, v in alloc["per_layer_bits"].items()}
    layers = parse_layers(args.layers)
    assert set(layers) <= set(range(24)), "layers must be within 0-23"
    os.makedirs(EVAL_CACHE, exist_ok=True)
    print(f"allocation bits: {[per_layer_bits[i] for i in layers]}", flush=True)
    print(f"expected_avg_bits={alloc['expected_avg_bits']:.4f}", flush=True)

    N_CAL, N_EVAL, T = 8, 4, 256
    cal, eva = build_seqs(N_CAL, N_EVAL, T)
    VOCAB = 151936

    lw = LazyWeights(SAFETENSORS)
    # cache unique embedding rows once (one pread/row, ~10MB total)
    uniq = sorted({int(x) for s in cal + eva for x in s})
    t0 = time.time()
    rows = lw.load_rows("model.embed_tokens.weight", uniq)
    mx.eval(rows)
    rowmap = dict(zip(uniq, [rows[i] for i in range(len(uniq))]))
    print(f"embed cache: {len(uniq)} rows in {time.time()-t0:.1f}s", flush=True)

    def embed_of(seq):
        h = mx.stack([rowmap[int(x)] for x in seq])
        mx.eval(h)
        return h

    qs = Qwen2Stream()
    final_norm = lw.load("model.norm.weight")
    mx.eval(final_norm)

    def load_block_fp(i: int):
        P = f"model.layers.{i}."
        W = {}
        for short in LIN_ORDER:
            W[short] = lw.load(P + WKEY[short])
        for short, full in BKEY.items():
            W[short] = lw.load(P + full)
        W["in_norm"] = lw.load(P + "input_layernorm.weight")
        W["post_norm"] = lw.load(P + "post_attention_layernorm.weight")
        mx.eval(*W.values())
        return W

    def shard_path(i: int):
        return os.path.join(EVAL_CACHE, f"layer_{i:02d}.npz")

    def ckpt_path(i: int | None = None):
        if i is None:
            return os.path.join(EVAL_CACHE, "hiddens_last.npz")
        return os.path.join(EVAL_CACHE, f"hiddens_{i:02d}.npz")

    # ---- FP perplexity on held-out eval seqs ---------------------------
    fp_ppl, fp_nll_tok = None, None
    if args.skip_fp and os.path.exists(EVAL_JSON):
        try:
            fp_ppl = json.load(open(EVAL_JSON)).get("fp_ppl")
            print(f"reusing fp_ppl={fp_ppl} from {EVAL_JSON}", flush=True)
        except Exception:
            fp_ppl = None
    if fp_ppl is None and not args.skip_fp:
        t0 = time.time()
        nll_sum, tok_sum = 0.0, 0
        for s in eva:
            h = embed_of(s)
            for i in range(24):
                W = load_block_fp(i)
                h = qs.block(h, W)
                mx.eval(h)
                del W
                mx.clear_cache()
            h = rmsnorm(h, final_norm)
            mx.eval(h)
            nll, n = seq_nll(h, s, lambda ii: lw.load_rows(
                "model.embed_tokens.weight", list(ii)), VOCAB,
                get_range=lambda a, b: lw.load_range(
                    "model.embed_tokens.weight", a, b))
            nll_sum += nll * n
            tok_sum += n
            del h
            mx.clear_cache()
        fp_nll_tok = nll_sum / tok_sum
        fp_ppl = float(np.exp(fp_nll_tok))
        print(f"FP ppl={fp_ppl:.3f} (nll={fp_nll_tok:.4f}, "
              f"{time.time()-t0:.1f}s)", flush=True)

    # ---- streaming quantize --------------------------------------------
    done = set()
    hiddens = None
    start_block = 0
    if args.resume:
        for i in layers:
            if os.path.exists(shard_path(i)):
                done.add(i)
        todo = [i for i in layers if i not in done]
        if todo:
            start_block = min(todo)
            ck = ckpt_path(start_block - 1) if start_block > 0 else None
            if ck is not None and os.path.exists(ck):
                z = np.load(ck)
                hiddens = {k: mx.array(z[k]).astype(mx.bfloat16)
                           for k in z.files}
                print(f"resume: {len(done)} shards, hiddens @ block "
                      f"{start_block - 1} -> starting at {start_block}",
                      flush=True)
            else:
                done = set()  # checkpoint missing: redo from scratch
                start_block = 0
                print("resume: ckpt missing, restarting", flush=True)

    t0 = time.time()
    q_times, hard_mse_log = {}, {}
    if hiddens is None:
        hiddens = {}
        for s_i, s in enumerate(cal):
            hiddens[f"cal{s_i}"] = embed_of(s)
        for s_i, s in enumerate(eva):
            hiddens[f"eva{s_i}"] = embed_of(s)
    cal_keys = [f"cal{s_i}" for s_i in range(N_CAL)]
    eva_keys = [f"eva{s_i}" for s_i in range(N_EVAL)]

    N_LAYERS_TOTAL = 24
    for i in range(start_block, N_LAYERS_TOTAL):
        if i in done:
            continue
        quantize_this = i in layers
        bits = per_layer_bits[i]
        Wfp = load_block_fp(i)
        # FP block forward; capture subsampled linear inputs for cal seqs
        caps = {name: [] for name in LIN_ORDER}
        for s_i, key in enumerate(cal_keys):
            h = hiddens[key]
            if quantize_this:
                out, cap = qs.block(h, Wfp, capture=True)
                mx.eval(out, *cap.values())
                for name in LIN_ORDER:
                    c = cap[name][::4]  # 64 toks/seq -> 512 total
                    mx.eval(c)
                    caps[name].append(c)
                del cap
            else:
                out = qs.block(h, Wfp)
                mx.eval(out)
            hiddens[key] = out
            del h
        for key in eva_keys:  # eval seqs ride along, no capture
            h = qs.block(hiddens[key], Wfp)
            mx.eval(h)
            hiddens[key] = h
        del Wfp
        mx.clear_cache()

        if quantize_this:
            grid = get_grid(bits)
            tB = time.time()
            Xstack = {n: mx.concatenate(caps[n], axis=0) for n in LIN_ORDER}
            mx.eval(*Xstack.values())
            del caps
            mx.clear_cache()
            shard = {"bits": np.array(bits), "grid": np.asarray(
                grid, dtype=np.float32)}
            for li, name in enumerate(LIN_ORDER):  # qk -> vo -> mlp order
                Wone = load_block_fp_single(lw, i, name)
                codes, scales, info = train_linear_mx(
                    [Xstack[name]], Wone, bits, args.steps,
                    args.resamples, seed=args.seed * 1000 + i * 10 + li)
                shard[f"{name}.codes"] = codes
                shard[f"{name}.scales"] = np.asarray(scales, dtype=np.float16)
                hard_mse_log[f"{i}.{name}"] = info.get("hard_mse", 0.0)
                del Wone
                mx.clear_cache()
            np.savez_compressed(shard_path(i), **shard)
            # re-forward through quantized block (frozen prefix for next)
            Wq = load_block_quant(shard, i, lw)
            for key in cal_keys + eva_keys:
                h = qs.block(hiddens[key], Wq)
                mx.eval(h)
                hiddens[key] = h
            del Wq, Xstack
            mx.clear_cache()
            q_times[i] = time.time() - tB
            done.add(i)
            print(f"layer {i} bits={bits} "
                  f"{q_times[i]:.1f}s mse={hard_mse_log.get(f'{i}.down_proj', 0):.5f}",
                  flush=True)
            per_layer = sum(q_times.values()) / max(1, len(q_times))
            print(f"  ETA remaining quant: "
                  f"{per_layer * (len(layers) - len(done)) / 60:.1f} min",
                  flush=True)
        # checkpoint hiddens every block (fp16, ~5MB, git-ignored dir)
        ck = {}
        for k, v in hiddens.items():
            ck[k] = np.asarray(v.astype(mx.float16))
        np.savez_compressed(ckpt_path(i), **ck)
        np.savez_compressed(ckpt_path(), **ck)
        del ck
        mx.clear_cache()

    quant_s = time.time() - t0

    # ---- quantized/hybrid perplexity ------------------------------------
    t0 = time.time()
    nll_sum, tok_sum = 0.0, 0
    for s_i, key in enumerate(eva_keys):
        h = rmsnorm(hiddens[key], final_norm)
        mx.eval(h)
        nll, n = seq_nll(h, eva[s_i], lambda ii: lw.load_rows(
            "model.embed_tokens.weight", list(ii)), VOCAB,
            get_range=lambda a, b: lw.load_range(
                "model.embed_tokens.weight", a, b))
        nll_sum += nll * n
        tok_sum += n
    quant_nll_tok = nll_sum / tok_sum
    quant_ppl = float(np.exp(quant_nll_tok))
    eval_s = time.time() - t0
    print(f"QUANT ppl={quant_ppl:.3f} (nll={quant_nll_tok:.4f}, "
          f"{eval_s:.1f}s)", flush=True)

    # ---- export ----------------------------------------------------------
    t0 = time.time()
    avg_bits = float(np.mean([per_layer_bits[i] for i in sorted(done)]))
    alloc_match = all(per_layer_bits[i] == alloc["per_layer_bits"][str(i)]
                      for i in sorted(done))
    with zipfile.ZipFile(FINAL_NPZ, "w", compression=zipfile.ZIP_STORED) as z:
        manifest = {"model": "Qwen2.5-0.5B", "group_size": 64,
                    "per_layer_bits": {str(i): per_layer_bits[i]
                                       for i in sorted(done)},
                    "grids": {}, "format": "uint8 codes + fp16 per-group "
                    "scales, row-major groups of 64 over input dim"}
        for i in sorted(done):
            sh = np.load(shard_path(i))
            manifest["grids"][str(i)] = [float(x) for x in sh["grid"]]
            for name in LIN_ORDER:
                for suf in ("codes", "scales"):
                    arr = sh[f"{name}.{suf}"]
                    buf = io.BytesIO()
                    np.lib.format.write_array(buf, np.ascontiguousarray(arr))
                    z.writestr(f"layer{i}.{name}.{suf}.npy", buf.getvalue())
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
    export_s = time.time() - t0

    total_s = time.time() - t_all
    partial = sorted(done) != list(range(24))
    remaining = [i for i in range(24) if i not in done]
    eta_s = (sum(q_times.values()) / max(1, len(q_times)) * len(remaining)
             if remaining else 0.0)
    out = {
        "fp_ppl": fp_ppl,
        "quant_ppl": quant_ppl,
        "delta_ppl": (quant_ppl - fp_ppl) if fp_ppl else None,
        "avg_bits": avg_bits,
        "expected_avg_bits": alloc["expected_avg_bits"],
        "allocation_match": bool(alloc_match),
        "per_layer_bits": {str(i): per_layer_bits[i] for i in sorted(done)},
        "layers_done": sorted(done),
        "partial": partial,
        "eta_remaining_s": eta_s,
        "runtime_s": {"total": total_s, "quantize": quant_s, "eval": eval_s,
                      "export": export_s, "per_layer": q_times},
        "config": {"steps": args.steps, "resamples": args.resamples,
                   "seed": args.seed, "group_size": 64, "max_tokens": 512,
                   "cal": f"{N_CAL}x{T} gsq.txt", "eval": f"{N_EVAL}x{T} rco.txt",
                   "compute": "bf16 fwd / fp32 Lion, batch-1, streamed"},
        "hard_mse": hard_mse_log,
        "honesty_note": ("SMOKE-SCALE, not paper-scale: 8x256 cal + 4x256 "
                         "held-out eval tokens (paper: 4096x4096), 20-50 Lion "
                         "steps/linear with 2-4 Gumbel resamples (paper: 20 "
                         "epochs), per-linear reconstruction losses in "
                         "qk->vo->mlp stage order (paper's joint "
                         "attention/block losses deferred), group_size 64 "
                         "(paper 128), norms/embeddings/biases kept fp."),
    }
    json.dump(out, open(EVAL_JSON, "w"), indent=1)
    print(f"wrote {EVAL_JSON} + {FINAL_NPZ} ({os.path.getsize(FINAL_NPZ)/1e6:.0f}MB)",
          flush=True)
    print(f"FP ppl={fp_ppl} QUANT ppl={quant_ppl} "
          f"delta={out['delta_ppl']} avg_bits={avg_bits:.3f} "
          f"total={total_s/60:.1f}min partial={partial}", flush=True)


def load_block_fp_single(lw, i: int, name: str):
    from gsq_mlx.e2e import WKEY
    import mlx.core as mx
    w = lw.load(f"model.layers.{i}." + WKEY[name])
    mx.eval(w)
    return w


def load_block_quant(shard: dict, i: int, lw):
    """Dequant one block from a shard dict (+fp biases/norms)."""
    import mlx.core as mx
    from gsq_mlx.e2e import dequant_linear, WKEY, BKEY
    import numpy as np
    grid = np.asarray(shard["grid"], dtype=np.float32)
    W = {}
    for short in ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"):
        W[short] = dequant_linear(shard[f"{short}.codes"],
                                  shard[f"{short}.scales"], grid)
    for short, full in BKEY.items():
        W[short] = lw.load(f"model.layers.{i}." + full)
    W["in_norm"] = lw.load(f"model.layers.{i}.input_layernorm.weight")
    W["post_norm"] = lw.load(f"model.layers.{i}.post_attention_layernorm.weight")
    mx.eval(*W.values())
    return W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synthetic",
                    choices=["synthetic", "real"])
    ap.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--layers", default="0-23")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--resamples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-fp", action="store_true")
    args = ap.parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
