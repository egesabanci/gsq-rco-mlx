"""RCO allocator for Qwen2.5-0.5B (OWNED by RCO side).

Two modes:
  --synthetic : Phase-1 stub. config.json shapes + synthetic sensitivities
                only; never touches model.safetensors. Fast.
  default     : Phase 2a. REAL per-layer sensitivity on a tiny cal set
                (<=8 seqs x 256 tokens, batch size 1, bf16/fp16, layer by
                layer with mx.clear_cache): RTN group_size=64 quantize of
                the 7 linears per layer for each option in [2,3,4,8], MSE
                vs FP on cal activations; cost = bits x param count. Then
                rco_loop over 24 groups, budget avg 3.0 bits, g=4,
                tau 1.0->0.01. Writes results/rco_allocation.json.

Memory rules: weights via mx.load only (NO torch); embedding dropped
after the FP pass; per-layer weights popped after scoring; FP acts spill
to a temp dir (fp16, ~458KB each), full activations never held.

Subsets: --layers 0-7 runs a partial pass (allocation over the subset +
full-run ETA) and waits for a go-ahead before --layers all.
--probe validates load + one-layer timing without the long run.

Usage:
  .venv/bin/python scripts/allocate_qwen.py --probe
  .venv/bin/python scripts/allocate_qwen.py --layers 0-7
  .venv/bin/python scripts/allocate_qwen.py --layers all
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models",
                         "Qwen2.5-0.5B")
CONFIG = os.path.join(MODEL_DIR, "config.json")
WEIGHTS = os.path.join(MODEL_DIR, "model.safetensors")
TOKENIZER = os.path.join(MODEL_DIR, "tokenizer.json")
OPTIONS = [2, 3, 4, 8]


def free_mb() -> float:
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=10).stdout
        for line in out.splitlines():
            if line.startswith("Pages free:"):
                return int(line.split(":")[1].strip().rstrip(".")
                           ) * 16384 / 1e6
    except Exception:
        pass
    return float("nan")


def parse_layers(spec: str, total: int) -> list:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(total))
    a, b = spec.split("-")
    layers = list(range(int(a), int(b) + 1))
    if not layers or min(layers) < 0 or max(layers) >= total:
        raise ValueError(f"--layers out of range: {spec}")
    return layers


# ------------------------------------------------------- synthetic path ---

def cmd_synthetic(args) -> None:
    from rco_mlx.rco import rco_loop
    total = 24
    Xo = np.array(OPTIONS, dtype=np.float64)
    N = total * 4
    rng = np.random.default_rng(args.seed)
    logits = rng.standard_normal((N, len(OPTIONS))) * 0.5
    res = rco_loop(logits, Xo, np.ones(N), args.avg_bits * N,
                   steps=args.steps, g=2, lr=args.lr, seed=args.seed)
    z = np.asarray(res["z_final"])
    per = (z.reshape(total, 4, -1) * np.array(OPTIONS)[None, None, :]
           ).sum(axis=(1, 2)) / 4
    print("synthetic per-layer avg bits:")
    for li, b in enumerate(per):
        print(f"  layer {li:2d} avg bits: {b:.2f}")
    print("Phase 1 synthetic stub OK (no model touched).")


# ----------------------------------------------------------- probe path ---

def cmd_probe(args) -> None:
    import mlx.core as mx
    from rco_mlx import phase2a as P
    T0 = time.time()
    cfg = P.load_cfg(args.config)
    print(f"cfg: {cfg} free={free_mb():.0f}MB")
    cal = P.build_cal_ids(args.tokenizer, 1, args.cal_tokens)
    print(f"cal: {cal.shape}, head ids {cal[0, :6].tolist()}")
    t0 = time.time()
    W = mx.load(args.weights)
    t_load = time.time() - t0
    print(f"mx.load: {len(W)} tensors in {t_load:.1f}s free={free_mb():.0f}MB")
    k0 = "model.layers.0.self_attn.q_proj.weight"
    print(f"layer0 q: {W[k0].shape} {W[k0].dtype}")
    Lw = P.split_layer(W, 0)
    n = sum(int(np.prod(W[k].shape)) for k in
            [f"model.layers.0.self_attn.{n}.weight"
             for n in ["q_proj", "k_proj", "v_proj", "o_proj"]]
            + [f"model.layers.0.mlp.{n}.weight"
               for n in ["gate_proj", "up_proj", "down_proj"]])
    print(f"layer0 quantized params: {n} (expect 14909440)")
    T = args.cal_tokens
    cos, sin = P.rope_cache(T, cfg["head_dim"], cfg["rope_theta"])
    mask = P.causal_mask(T)
    ids = mx.array(cal[0].astype(np.int32))
    x = mx.take(W["model.embed_tokens.weight"], ids, axis=0)
    mx.eval(x)
    t0 = time.time()
    y = P.layer_forward(x, Lw, cfg, cos, sin, mask)
    mx.eval(y)
    t_fp = time.time() - t0
    print(f"layer FP forward ({T}x{cfg['hidden']} bf16): {t_fp:.2f}s")
    for bits in OPTIONS:
        t0 = time.time()
        Q = P.quant_layer(Lw, bits)
        mx.eval(*[Q[k] for k in ["q_w", "k_w", "v_w", "o_w", "gate_w",
                                 "up_w", "down_w"]])
        yq = P.layer_forward(x, Q, cfg, cos, sin, mask)
        m = mx.mean(mx.square((yq - y).astype(mx.float32)))
        mx.eval(m)
        print(f"  bits={bits} quant+fwd+mse: {time.time()-t0:.2f}s "
              f"mse={float(m.item()):.6e}")
        del Q, yq, m
        mx.clear_cache()
    # extrapolate: collect(8*24 fwd) + sens(24 layers * (8 FP equiv + 4*8 Q))
    unit = t_fp
    est = 8 * 24 * unit + 24 * 8 * 4 * unit * 1.5 + t_load
    print(f"rough full-pass estimate: ~{est/60:.1f} min (probe {time.time()-T0:.0f}s)")
    del W, x, y
    mx.clear_cache()
    gc.collect()


# ------------------------------------------------------------ real path ---

def cmd_real(args) -> None:
    import mlx.core as mx
    from rco_mlx import phase2a as P
    T0 = time.time()
    if args.cal_seqs > 8 or args.cal_tokens > 256:
        raise SystemExit("memory rule: calibration capped at 8x256")
    cfg = P.load_cfg(args.config)
    layers = parse_layers(args.layers, cfg["layers"])
    partial = len(layers) != cfg["layers"]
    print(f"Qwen2.5-0.5B real pass: layers={args.layers} "
          f"({len(layers)} layers){' PARTIAL' if partial else ''}, "
          f"cal={args.cal_seqs}x{args.cal_tokens}, free={free_mb():.0f}MB")

    cal_ids = P.build_cal_ids(args.tokenizer, args.cal_seqs,
                              args.cal_tokens)
    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="rco_cal_")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"act cache: {cache_dir}")

    t0 = time.time()
    W = mx.load(args.weights)
    t_load = time.time() - t0
    print(f"mx.load {len(W)} tensors: {t_load:.1f}s free={free_mb():.0f}MB")

    T = args.cal_tokens
    cos, sin = P.rope_cache(T, cfg["head_dim"], cfg["rope_theta"])
    mask = P.causal_mask(T)

    # FP collection over the prefix needed (0..max(layers)), batch size 1.
    need = list(range(max(layers) + 1))
    marker = os.path.join(cache_dir, "collection.json")
    have = []
    if os.path.exists(marker):
        try:
            meta = json.load(open(marker))
            if (meta.get("seqs") == args.cal_seqs
                    and meta.get("tokens") == T):
                have = [i for i in need if os.path.exists(
                    os.path.join(cache_dir, f"out_{i}_{args.cal_seqs-1}.npy"))]
        except Exception:
            pass
    todo = [i for i in need if i not in have]
    if todo:
        print(f"collecting FP acts for layers {todo[0]}..{todo[-1]} "
              f"({len(todo)} layers x {args.cal_seqs} seqs)")
        embed = W.pop("model.embed_tokens.weight")
        info = P.collect_fp(W, embed, cfg, cal_ids, todo, cache_dir,
                            cos, sin, mask)
        del embed
        mx.clear_cache()
        gc.collect()
        json.dump({"seqs": args.cal_seqs, "tokens": T,
                   "layers": sorted(set(have) | set(todo))},
                  open(marker, "w"))
        print(f"collection: {info['seconds']:.1f}s free={free_mb():.0f}MB")
    else:
        print("reusing cached FP acts")
    # embedding no longer needed even if it survived (cache-reuse path)
    W.pop("model.embed_tokens.weight", None)
    mx.clear_cache()

    counts = np.array([P.layer_param_count(cfg)] * cfg["layers"])
    S = np.zeros((len(layers), len(OPTIONS)))
    sens_t = []
    for li, i in enumerate(layers):
        t0 = time.time()
        ms = P.score_layer(W, cfg, cache_dir, i, args.cal_seqs,
                           cos, sin, mask, OPTIONS)
        dt = time.time() - t0
        sens_t.append(dt)
        for kj, b in enumerate(OPTIONS):
            S[li, kj] = ms[str(b)]["mean"]
        print(f"layer {i:2d} mse " +
              " ".join(f"{b}b:{ms[str(b)]['mean']:.3e}"
                       for b in OPTIONS) +
              f" [{dt:.1f}s] free={free_mb():.0f}MB")
        if len(sens_t) >= 2:
            eta = sum(sens_t) / len(sens_t) * (cfg["layers"] - (li + 1))
            print(f"    ETA remaining layers at this rate: ~{eta/60:.1f} min")

    alloc = P.run_allocation(S, counts[layers], args.avg_bits,
                             args.steps, args.samples, args.lr, args.seed,
                             init=args.init)
    gap = alloc["budget_gap"]
    ok_budget = gap < 1e-4
    ok_decode = alloc["res"]["decoded_cost"] <= alloc["res"]["budget_int"]
    print(f"budget: C={alloc['res']['history'][-1]['cost']:.6f} "
          f"B={alloc['budget']:.6f} |C-B|={gap:.2e} "
          f"[{'PASS' if ok_budget else 'FAIL'}]")
    print(f"decode: {alloc['res']['decoded_cost']} <= "
          f"{alloc['res']['budget_int']} [{'PASS' if ok_decode else 'FAIL'}]")
    print("allocation table (layer | params | mse 2/3/4/8 | bits):")
    for li, i in enumerate(layers):
        print(f"  layer {i:2d} | {counts[i]/1e6:5.2f}M | " +
              " ".join(f"{v:.2e}" for v in S[li]) +
              f" | {alloc['per_layer_bits'][li]:.0f}b")
    print(f"expected avg bits: {alloc['expected_avg_bits']:.4f} "
          f"(target {args.avg_bits})")
    print(f"decoded avg bits: "
          f"{alloc['res']['decoded_cost'] / alloc['weights_m'].sum():.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rec = {
        "partial": partial,
        "layers": layers,
        "dims": cfg,
        "options": OPTIONS,
        "group_size": 64,
        "calibration": {"seqs": args.cal_seqs, "tokens": T, "batch": 1,
                        "dtype": "bf16(compute)/fp16(spill)"},
        "quantized_params_per_layer": int(counts[0]),
        "excluded_from_cost": ["model.embed_tokens", "norms", "biases"],
        "sensitivities_mse": {str(i): [float(v) for v in row]
                              for i, row in zip(layers, S)},
        "per_layer_bits": {str(i): int(b) for i, b in
                           zip(layers, alloc["per_layer_bits"])},
        "expected_avg_bits": alloc["expected_avg_bits"],
        "budget": alloc["budget"],
        "budget_gap": gap,
        "budget_ok": bool(ok_budget),
        "decoded_cost_int": int(alloc["res"]["decoded_cost"]),
        "budget_int": int(alloc["res"]["budget_int"]),
        "decode_ok": bool(ok_decode),
        "rco": {"steps": args.steps, "g": args.samples, "lr": args.lr,
                "tau": "1.0->0.01 exp", "seed": args.seed,
                "init": args.init,
                "upstream": "linear expected-sensitivity surrogate",
                "history": alloc["res"]["history"]},
        "timing": {"load_s": t_load,
                   "sens_s_per_layer": sens_t,
                   "total_s": time.time() - T0},
        "notes": ("Per-sample quantized forwards deferred (10x4x8 "
                  "full-model too heavy); gradient = normalized "
                  "sensitivity table, DP sampled per g-sample per Alg.1. "
                  "Loop is a local refiner at lr 0.1/10 steps (logits move "
                  "~1.5, decode cells hold): greedy DP(-S) init decodes to "
                  "totS 1.33 vs 5.60 from soft -Sn init; shipped = greedy "
                  "init + RCO refine/verify."),
    }
    json.dump(rec, open(args.out, "w"), indent=1)
    print(f"wrote {args.out} total {time.time()-T0:.1f}s")
    if not args.keep_cache and not args.cache_dir:
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)
    if not (ok_budget and ok_decode):
        raise SystemExit("budget verification FAILED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--avg-bits", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--init", default="greedy", choices=["greedy", "negS"])
    ap.add_argument("--cal-seqs", type=int, default=8)
    ap.add_argument("--cal-tokens", type=int, default=256)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--keep-cache", action="store_true")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "rco_allocation.json"))
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()
    if args.synthetic:
        cmd_synthetic(args)
    elif args.probe:
        cmd_probe(args)
    else:
        cmd_real(args)


if __name__ == "__main__":
    main()
