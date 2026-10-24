"""GSQ quantize for Qwen2.5-0.5B (OWNED by GSQ side).

Modes:
  synthetic (default, Phase-1 stub): shapes from config.json + synthetic
      weights only. Never touches model.safetensors.
      Usage: .venv/bin/python scripts/quantize_qwen.py [--bits 2]
  real (block-wise staged GSQ, paper Sec 3.3 / 4.1 / App C, H): quantizes the
      real checkpoint one Transformer block at a time (gsq_mlx.staged).
      Calibration: WikiText-2 train windows (gsq_rco_common.data). Every block
      is trained on the quantized prefix x_q against the FP block applied to
      x_q, and then x_q <- Q_block(x_q), so each block is applied once per
      stream. Outputs under --out (default results/.gsq_cache): one shard
      layer_XX.npz per block (per linear codes uint8, scales fp16, bits; the
      stats of the block as JSON), run_log.json, and the mlx-lm checkpoint in
      <out>/mlx (gsq_mlx.export). --resume replays the quantized prefix from
      the shards, so no block is skipped in the stream.
      Usage: .venv/bin/python scripts/quantize_qwen.py --mode real
             --alloc results/rco_allocation.json [--layers 0-23] [--steps 200]
             (or --uniform-bits 2) [--resume] [--scale-finetune-steps N]

NEVER touches src/rco_mlx/* or scripts/allocate_qwen.py.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
CONFIG = os.path.join(MODEL_DIR, "config.json")
SAFETENSORS = os.path.join(MODEL_DIR, "model.safetensors")
OUT_DIR = os.path.join(ROOT, "results", ".gsq_cache")
N_LAYERS = 24


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


def load_bits(args) -> dict:
    """{layer: {linear: bits}} for the layers to quantize, from --alloc or --uniform-bits.
    --alloc JSON: {"per_linear_bits": {"<layer>.<linear>": b}} or {"per_layer_bits": {"<layer>": b}}."""
    from gsq_mlx.e2e import LIN_ORDER
    layers = parse_layers(args.layers)
    if (args.alloc is None) == (args.uniform_bits is None):
        raise SystemExit("give exactly one of --alloc and --uniform-bits")
    if args.uniform_bits is not None:
        per = {(i, n): args.uniform_bits for i in layers for n in LIN_ORDER}
    else:
        alloc = json.load(open(args.alloc))
        if "per_linear_bits" in alloc:
            per = {}
            for key, b in alloc["per_linear_bits"].items():
                i, name = key.split(".", 1)
                per[(int(i), name)] = int(b)
        elif "per_layer_bits" in alloc:
            per = {(i, n): int(alloc["per_layer_bits"][str(i)]) for i in layers for n in LIN_ORDER}
        else:
            raise SystemExit(f"{args.alloc}: needs per_linear_bits or per_layer_bits")
    out = {}
    for i in layers:
        missing = [n for n in LIN_ORDER if (i, n) not in per]
        if missing:
            raise SystemExit(f"layer {i} has no bits for {missing}")
        out[i] = {n: per[(i, n)] for n in LIN_ORDER}
        for n, b in out[i].items():
            if not 1 <= b <= 8:
                raise SystemExit(f"layer {i} {n}: bits {b} outside 1..8")
    return out


def fingerprint(args) -> dict:
    """Settings that must match for shards to be reused on resume."""
    from gsq_mlx.lion import LR_SCALES
    return {"model": "Qwen2.5-0.5B", "group_size": args.group_size, "damp": args.damp,
            "block_size": 128, "steps": args.steps, "micro_batch": args.micro_batch,
            "resamples": args.resamples, "lr_logits": args.lr_logits, "lr_scales": LR_SCALES,
            "seed": args.seed, "n_cal": args.n_cal, "seq_len": args.seq_len,
            "cal_seed": args.cal_seed}


def fp_block_weights(lw, i: int) -> dict:
    """FP weights of block i (linears, q/k/v biases, norms), bf16."""
    import mlx.core as mx
    from gsq_mlx.e2e import BKEY, WKEY
    P = f"model.layers.{i}."
    W = {short: lw.load(P + full) for short, full in WKEY.items()}
    W.update({short: lw.load(P + full) for short, full in BKEY.items()})
    W["in_norm"] = lw.load(P + "input_layernorm.weight")
    W["post_norm"] = lw.load(P + "post_attention_layernorm.weight")
    mx.eval(*W.values())
    return W


def save_shard(path: str, layer: int, lins: dict, bits: dict, fp: dict, stats: dict) -> None:
    arrays = {"meta": np.array(json.dumps({"layer": layer, "fingerprint": fp,
                                           "bits": bits})),
              "stats": np.array(json.dumps(stats))}
    for name, q in lins.items():
        arrays[f"{name}.codes"] = q["codes"]
        arrays[f"{name}.scales"] = q["scales"]
        arrays[f"{name}.bits"] = np.array(q["bits"])
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_shard(path: str):
    """-> (meta dict, stats dict, {linear: {codes, scales, bits}}), fully read into memory."""
    from gsq_mlx.e2e import LIN_ORDER
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        stats = json.loads(str(z["stats"]))
        lins = {n: {"codes": z[f"{n}.codes"], "scales": z[f"{n}.scales"],
                    "bits": int(z[f"{n}.bits"])} for n in LIN_ORDER}
    return meta, stats, lins


def _check_meta(meta: dict, fp: dict, bits: dict, path: str) -> None:
    if meta["fingerprint"] != fp or {k: int(v) for k, v in meta["bits"].items()} != bits:
        raise SystemExit(f"{path} was made with other settings; delete it or change the arguments")


def run_real(args) -> None:
    import mlx.core as mx
    from gsq_rco_common.data import calibration_windows, eval_windows
    from gsq_rco_common.evaluate import nll_from_hidden
    from gsq_mlx import staged
    from gsq_mlx.e2e import BKEY, LIN_ORDER, WKEY, LazyWeights, Qwen2Stream, rmsnorm
    from gsq_mlx.export import bits_report, export_mlx_lm
    from gsq_mlx.lion import LR_SCALES

    t_all = time.perf_counter()
    os.makedirs(args.out, exist_ok=True)
    bits_by_layer = load_bits(args)
    layers = sorted(bits_by_layer)
    fp = fingerprint(args)
    opts = staged.BlockOpts(group_size=args.group_size, damp=args.damp, steps=args.steps,
                            micro_batch=args.micro_batch, resamples=args.resamples,
                            lr_logits=args.lr_logits, seed=args.seed)
    shard_path = lambda i: os.path.join(args.out, f"layer_{i:02d}.npz")
    done = {i for i in layers if os.path.exists(shard_path(i))}
    if done and not args.resume:
        raise SystemExit(f"{len(done)} shard(s) already in {args.out}: pass --resume to reuse them")
    for i in sorted(done):
        _check_meta(load_shard(shard_path(i))[0], fp, bits_by_layer[i], shard_path(i))

    cal = calibration_windows(args.n_cal, args.seq_len, seed=args.cal_seed)   # int32 [N, T]
    lw = LazyWeights(SAFETENSORS)
    stream = Qwen2Stream()
    h = mx.stack([lw.load_rows("model.embed_tokens.weight", list(s)) for s in cal])
    mx.eval(h)
    print(f"calibration {cal.shape} from WikiText-2 train; layers {layers}; "
          f"bits {[sorted(set(bits_by_layer[i].values())) for i in layers]}", flush=True)

    # ---- quantize block by block on the quantized prefix ----------------
    layer_stats = {}
    for i in range(N_LAYERS):
        t0 = time.perf_counter()
        Wfp = fp_block_weights(lw, i)
        if i not in bits_by_layer:
            Wq = Wfp                                   # not quantized: FP block in the stream
        elif i in done:
            meta, stats, lins = load_shard(shard_path(i))
            Wq = staged.hard_block(lins, {k: Wfp[k] for k in Wfp if k not in LIN_ORDER},
                                   args.group_size)
            layer_stats[i] = stats
            print(f"layer {i}: replayed from {shard_path(i)}", flush=True)
        else:
            print(f"layer {i}: bits {bits_by_layer[i]}", flush=True)
            shard, stats, Wq = staged.quantize_block(stream, h, Wfp, bits_by_layer[i], opts, i)
            stats["seconds"] = round(time.perf_counter() - t0, 2)
            save_shard(shard_path(i), i, shard, bits_by_layer[i], fp, stats)
            layer_stats[i] = stats
            print(f"layer {i}: done in {stats['seconds']:.1f}s, peak MLX "
                  f"{stats['peak_mlx_bytes'] / 2**30:.2f} GB, max RSS "
                  f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f} GB",
                  flush=True)
        h = staged.advance(stream, h, Wq)
        del Wfp, Wq
        mx.clear_cache()
    del h
    mx.clear_cache()

    # ---- optional scale-only end-to-end fine-tune (App H) ---------------
    fine = None
    if args.scale_finetune_steps > 0:
        t0 = time.perf_counter()
        E = lw.load("model.embed_tokens.weight")
        fnorm = lw.load("model.norm.weight")
        hf = mx.stack([lw.load_rows("model.embed_tokens.weight", list(s)) for s in cal])
        for i in range(N_LAYERS):
            hf = staged.advance(stream, hf, fp_block_weights(lw, i))
        h_ref = rmsnorm(hf, fnorm)
        mx.eval(h_ref)
        del hf
        qblocks, base = {}, {}
        for i in range(N_LAYERS):
            Wfp = fp_block_weights(lw, i)
            if i in bits_by_layer:
                _, _, lins = load_shard(shard_path(i))
                qblocks[i] = lins
                base[i] = {k: Wfp[k] for k in Wfp if k not in LIN_ORDER}
            else:
                base[i] = Wfp
        new, kls = staged.scale_finetune(stream, qblocks, base, E, fnorm, cal, h_ref,
                                         args.scale_finetune_steps, LR_SCALES,
                                         args.group_size)
        w = max(1, min(10, len(kls) // 2))  # averaging window: at most half the steps
        for i in layers:
            meta, stats, lins = load_shard(shard_path(i))
            for n in LIN_ORDER:
                lins[n]["scales"] = new[i][n]
            stats["scale_finetune"] = {"steps": args.scale_finetune_steps, "window": w,
                                       "kl_first": float(np.mean(kls[:w])),
                                       "kl_last": float(np.mean(kls[-w:]))}
            save_shard(shard_path(i), i, lins, bits_by_layer[i], fp, stats)
            layer_stats[i] = stats
        fine = {"steps": args.scale_finetune_steps, "kl_first": float(np.mean(kls[:w])),
                "kl_last": float(np.mean(kls[-w:])), "seconds": round(time.perf_counter() - t0, 1)}
        print(f"scale fine-tune: KL {fine['kl_first']:.5f} -> {fine['kl_last']:.5f} "
              f"in {fine['seconds']:.1f}s", flush=True)
        del E, fnorm, h_ref, qblocks, base
        mx.clear_cache()

    # ---- sanity perplexity on validation windows (not the full evaluation) ------
    sanity = None
    if args.sanity_windows > 0:
        ev = eval_windows(args.seq_len, split="validation", max_windows=args.sanity_windows)
        E = lw.load("model.embed_tokens.weight")
        fnorm = lw.load("model.norm.weight")
        head = staged.tied_head(E)
        hq = mx.stack([lw.load_rows("model.embed_tokens.weight", list(s)) for s in ev])
        hf = hq
        for i in range(N_LAYERS):
            Wfp = fp_block_weights(lw, i)
            if i in bits_by_layer:
                _, _, lins = load_shard(shard_path(i))
                Wq = staged.hard_block(lins, {k: Wfp[k] for k in Wfp if k not in LIN_ORDER},
                                       args.group_size)
            else:
                Wq = Wfp
            hq = staged.advance(stream, hq, Wq)
            hf = staged.advance(stream, hf, Wfp)
            del Wfp, Wq
            mx.clear_cache()
        sums = {}
        for name, hs in (("fp", hf), ("quant", hq)):
            hn = rmsnorm(hs, fnorm)
            tot, n = nll_from_hidden(head, hn[:, :-1], ev[:, 1:])
            sums[name] = (float(tot), n)
        sanity = {"windows": int(ev.shape[0]), "seq_len": int(ev.shape[1]),
                  "fp_ppl": math.exp(sums["fp"][0] / sums["fp"][1]),
                  "quant_ppl": math.exp(sums["quant"][0] / sums["quant"][1]),
                  "note": "sanity only (validation windows); the orchestrator runs evaluate.perplexity"}
        print(f"sanity ppl on {sanity['windows']}x{sanity['seq_len']} validation: "
              f"FP {sanity['fp_ppl']:.3f} quantized {sanity['quant_ppl']:.3f}", flush=True)
        del E, fnorm, head, hq, hf
        mx.clear_cache()

    # ---- mlx-lm export of the quantized linears ----------------------------
    export = None
    if not args.no_export:
        quantized, n_params = {}, 0
        for i in layers:
            _, _, lins = load_shard(shard_path(i))
            for name in LIN_ORDER:
                q = lins[name]
                path = f"model.layers.{i}." + WKEY[name][: -len(".weight")]
                quantized[path] = dict(codes=q["codes"], scales=q["scales"].astype(np.float32),
                                       bits=q["bits"], group_size=args.group_size)
                n_params += q["codes"].size
        export_dir = args.export_dir or os.path.join(args.out, "mlx")
        export_mlx_lm(export_dir, quantized, MODEL_DIR)
        rep = bits_report(quantized, n_params)
        export = {"dir": export_dir, "linears": len(quantized),
                  "bits_per_weight": {k: rep[k] for k in ("code_bpp", "native_bpp", "mlx_bpp",
                                                          "shard_bpp")}}
        print(f"exported {len(quantized)} linears -> {export_dir}: code "
              f"{rep['code_bpp']:.3f} | native {rep['native_bpp']:.3f} | MLX "
              f"{rep['mlx_bpp']:.3f} bits/weight", flush=True)

    run = {"config": dict(fp, layers=layers, uniform_bits=args.uniform_bits, alloc=args.alloc,
                          layers_arg=args.layers, scale_finetune_steps=args.scale_finetune_steps),
           "layers": {str(i): layer_stats[i] for i in layers if i in layer_stats},
           "scale_finetune": fine, "sanity": sanity, "export": export,
           "seconds_total": round(time.perf_counter() - t_all, 1),
           "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    with open(os.path.join(args.out, "run_log.json"), "w") as f:
        json.dump(run, f, indent=1)
    print(f"wrote {os.path.join(args.out, 'run_log.json')} in {run['seconds_total'] / 60:.1f} min; "
          f"max RSS {run['max_rss_bytes'] / 2**30:.2f} GB", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    ap.add_argument("--bits", type=int, default=2, choices=[1, 2, 3, 4])
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--layers", default="0-23")
    ap.add_argument("--alloc", help="allocation JSON (per_linear_bits or per_layer_bits)")
    ap.add_argument("--uniform-bits", type=int, choices=range(1, 9))
    ap.add_argument("--n-cal", type=int, default=128, help="calibration sequences")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--cal-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=200, help="Lion steps per stage")
    ap.add_argument("--lr-logits", type=float, default=1e-4, help="logit lr (Table 7)")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--resamples", type=int, default=2, help="Gumbel resamples per step")
    ap.add_argument("--damp", type=float, default=0.01, help="GPTQ damping")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT_DIR, help="shard, log and export directory")
    ap.add_argument("--export-dir", help="mlx-lm export directory (default <out>/mlx)")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--sanity-windows", type=int, default=2,
                    help="validation windows for the sanity perplexity (0 to skip)")
    ap.add_argument("--scale-finetune-steps", type=int, default=0, help="App H steps (0 = off)")
    ap.add_argument("--resume", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "synthetic":
        run_synthetic(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()
