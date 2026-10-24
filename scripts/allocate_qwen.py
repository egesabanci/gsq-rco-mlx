"""RCO bit allocation for Qwen2.5-0.5B on the quantized database (WP6, paper App B and Alg.1).

Subcommands (run one at a time under the shared memory lock, RULES item 4):
  build-db   quantize every linear at every option with the deployment quantizer (GPTQ with
             signed scales by default), store codes, scales and per-linear output-MSE
             sensitivities. Hessians come from WikiText-2 train windows of the FP model.
  allocate   RCO (Alg.1) with the KL objective (Eq.6) over the database. Writes the
             allocation JSON (--out is required).
  evaluate   WikiText-2 test perplexity of the FP model, of a uniform allocation, of the
             proxy-DP baseline, or of an allocation JSON written by `allocate`.

Examples:
  heavy.sh .venv/bin/python scripts/allocate_qwen.py build-db --db-dir <dir>
  heavy.sh .venv/bin/python scripts/allocate_qwen.py allocate --db-dir <dir> --out <file.json>
  heavy.sh .venv/bin/python scripts/allocate_qwen.py evaluate --db-dir <dir> --uniform 3
The default database directory is results/.rco_db (git-ignored). results/rco_allocation.json is
written only by a full 24-layer run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO / "results" / ".rco_db"
DEFAULT_OUT = REPO / "results" / "rco_allocation.json"
N_LAYERS = 24
BUDGET_TOL = 1e-4          # |C - B| per step, the acceptance bound of the brief
SCALE_BITS_PER_WEIGHT = 16  # one fp16 scale per group; bpp = code bits + 16 / group_size


def model_fingerprint() -> str:
    from gsq_rco_common.data import MODEL_DIR
    from rco_mlx.database import sha256_files
    return sha256_files([MODEL_DIR / "config.json", MODEL_DIR / "model.safetensors"])


def calibration_tokens(seqs: int, length: int, seed: int, split: str) -> np.ndarray:
    from gsq_rco_common.data import calibration_windows
    return calibration_windows(seqs, length, seed=seed, split=split)


def load_model(n_layers: int, lazy: bool = False):
    """The FP Qwen2.5-0.5B cut to n_layers blocks. lazy=True (build-db only) leaves every
    weight unloaded until its block is used, through mlx_lm load_model(lazy=True); the other
    commands need all their blocks resident and use gsq_rco_common.model.load_fp_model."""
    if lazy:
        from gsq_rco_common.data import MODEL_DIR
        from mlx_lm.utils import load_model as mlx_load_model
        model, _ = mlx_load_model(MODEL_DIR, lazy=True)
        return model
    from gsq_rco_common.model import load_fp_model
    from rco_mlx.allocate import truncate_layers
    return truncate_layers(load_fp_model(), n_layers)


def cmd_build_db(args) -> None:
    from rco_mlx.database import DBSpec, build_database
    t0 = time.time()
    spec = DBSpec(options=args.options, group_size=args.group_size, quantizer=args.db_quantizer,
                  damp=args.damp, n_layers=args.n_layers, calib_split=args.calib_split,
                  calib_seed=args.calib_seed)
    tokens = calibration_tokens(args.calib_seqs, args.calib_len, args.calib_seed, args.calib_split)
    print(f"[build-db] quantizer={spec.quantizer} options={spec.options} "
          f"group={spec.group_size} damp={spec.damp} layers={spec.n_layers} "
          f"calibration={tokens.shape[0]}x{tokens.shape[1]} "
          f"({tokens.shape[0] * (tokens.shape[1] - 1)} Hessian rows) -> {args.db_dir}", flush=True)
    model = load_model(args.n_layers, lazy=True)
    print(f"[build-db] model opened lazily in {time.time() - t0:.1f}s", flush=True)
    manifest = build_database(model, tokens, spec, args.db_dir, model_fingerprint(), log=print,
                              release=True)
    print(f"[build-db] wrote {Path(args.db_dir) / 'manifest.json'} "
          f"({len(manifest['groups'])} linears) in {time.time() - t0:.1f}s", flush=True)


def _open_database(db_dir, n_layers: int, model_sha: str):
    from rco_mlx.database import Database
    db = Database(db_dir)
    c = db.manifest["calibration"]
    db.verify(model_sha, calibration_tokens(c["n_windows"], c["window_len"], c["seed"], c["split"]))
    if n_layers != db.n_layers:
        raise SystemExit(f"--n-layers {n_layers} but the database has {db.n_layers} layers")
    return db


def cmd_allocate(args) -> None:
    from gsq_rco_common.evaluate import CHUNK
    from rco_mlx.allocate import KLObjective, allocation_summary, rco_allocate
    from rco_mlx.rco import integer_weights
    if args.out is None:
        raise SystemExit("--out is required")
    out = Path(args.out).resolve()
    if out == DEFAULT_OUT.resolve() and args.n_layers != N_LAYERS:
        raise SystemExit("results/rco_allocation.json is written only by the full 24-layer run; "
                         "pass another --out for partial runs")
    t0 = time.time()
    sha = model_fingerprint()
    db = _open_database(args.db_dir, args.n_layers, sha)
    model = load_model(args.n_layers)
    kl_tokens = calibration_tokens(args.kl_seqs, args.kl_len, args.kl_seed, "train")
    objective = KLObjective(model, db, kl_tokens, windows_per_sample=args.windows_per_sample)
    setup_s = time.time() - t0
    print(f"[allocate] {len(db.groups)} linears, options {db.options}, avg {args.avg_bits} bits, "
          f"steps={args.steps} g={args.g} lr={args.lr} KL windows {kl_tokens.shape} "
          f"setup {setup_s:.1f}s", flush=True)

    last = [time.time()]

    def callback(t, logits, record):
        now = time.time()
        record["seconds"] = now - last[0]
        last[0] = now
        print(f"[allocate] step {t:3d} tau={record['tau']:.4f} cost={record['cost']:.6f} "
              f"loss={record.get('loss', float('nan')):.5f} {record['seconds']:.1f}s", flush=True)

    res = rco_allocate(objective, db, args.avg_bits, steps=args.steps, g=args.g, lr=args.lr,
                       seed=args.seed, tau0=args.tau0, tau_min=args.tau_min, init=args.init,
                       callback=callback)
    per_linear, per_layer = allocation_summary(db, res["choice"])
    gaps = [abs(h["cost"] - res["budget"]) for h in res["history"]]
    budget_ok = max(gaps) < BUDGET_TOL
    decode_ok = res["decoded_int_cost"] <= res["budget_int"]
    unit = int(integer_weights(db.counts)[1])
    code_bits = res["decoded_avg_bits"]
    rec = {
        "method": "RCO (Alg.1), exact budget, KL objective (App B Eq.6), STE upstream per linear",
        "config": {"avg_bits": args.avg_bits, "steps": args.steps, "g": args.g, "lr": args.lr,
                   "seed": args.seed, "tau0": args.tau0, "tau_min": args.tau_min,
                   "init": args.init, "n_layers": args.n_layers,
                   "windows_per_sample": args.windows_per_sample,
                   "kl_seqs": args.kl_seqs, "kl_len": args.kl_len, "kl_seed": args.kl_seed,
                   "kl_chunk": CHUNK},
        "database": {"dir": str(Path(args.db_dir).resolve()), "options": list(db.options),
                     "group_size": db.group_size, "quantizer": db.manifest["quantizer"],
                     "damp": db.manifest["damp"], "n_layers": db.n_layers,
                     "identity": {k: db.manifest[k] for k in
                                  ("format", "model_weights_sha256", "calibration", "hessian")}},
        "budget": {"avg_bits_target": args.avg_bits, "budget_units": res["budget"],
                   "budget_int": res["budget_int"], "ticks": res["ticks"],
                   "total_units": res["total_units"], "unit_params": unit},
        "expected_avg_bits": res["expected_avg_bits"],
        "decoded_int_cost": res["decoded_int_cost"],
        "decoded_avg_bits": code_bits,
        "bpp_avg_incl_scales": code_bits + SCALE_BITS_PER_WEIGHT / db.group_size,
        "checks": {"max_abs_budget_gap": max(gaps), "budget_tol": BUDGET_TOL,
                   "budget_ok": bool(budget_ok), "decode_ok": bool(decode_ok)},
        "per_linear_bits": per_linear,
        "per_layer": per_layer,
        "history": res["history"],
        "timing": {"setup_s": setup_s,
                   "seconds_per_step_mean": float(np.mean([h["seconds"] for h in res["history"]])),
                   "total_s": time.time() - t0},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[allocate] expected avg {res['expected_avg_bits']:.6f} bits, decoded "
          f"{res['decoded_int_cost']}/{res['budget_int']} -> {code_bits:.6f} code bits, "
          f"{rec['bpp_avg_incl_scales']:.6f} bpp incl. scales", flush=True)
    print(f"[allocate] budget max|C-B|={max(gaps):.2e} [{'PASS' if budget_ok else 'FAIL'}] "
          f"decode [{'PASS' if decode_ok else 'FAIL'}] wrote {out} total {time.time() - t0:.1f}s",
          flush=True)
    if not (budget_ok and decode_ok):
        raise SystemExit("budget verification FAILED")


def cmd_evaluate(args) -> None:
    from gsq_rco_common.data import eval_windows
    from gsq_rco_common.evaluate import perplexity
    from rco_mlx.allocate import (decoded_cost, install_choice, proxy_dp_choice,
                                  uniform_choice)
    chosen =[args.fp, args.uniform is not None, args.proxy_dp, args.allocation is not None]
    if sum(chosen) != 1:
        raise SystemExit("choose exactly one of --fp, --uniform, --proxy-dp, --allocation")
    t0 = time.time()
    choice = None
    db = None
    if not args.fp:
        db = _open_database(args.db_dir, args.n_layers, model_fingerprint())
    if args.uniform is not None:
        mode, choice = f"uniform-{args.uniform}", uniform_choice(db, args.uniform)
    elif args.proxy_dp:
        mode, choice = f"proxy-dp-{args.avg_bits}", proxy_dp_choice(db, args.avg_bits)
    elif args.allocation is not None:
        with open(args.allocation) as f:
            saved = json.load(f)
        identity = saved["database"]["identity"]
        if identity != {k: db.manifest[k] for k in identity}:
            raise SystemExit(f"{args.allocation} was made from a different database")
        bits = saved["per_linear_bits"]
        choice = np.array([db.options.index(bits[f"{L}.{n}"]) for L, n in db.groups])
        mode = f"allocation:{Path(args.allocation).name}"
    else:
        mode = "fp"
    model = load_model(args.n_layers)
    if choice is not None:
        install_choice(model, db, choice)
        cost, avg = decoded_cost(db, choice)
    else:
        cost, avg = None, None
    windows = eval_windows(seq_len=args.eval_len, split="test", max_windows=args.eval_windows)
    print(f"[evaluate] {mode}: {windows.shape[0]} WikiText-2 test windows x {windows.shape[1]} "
          f"tokens, setup {time.time() - t0:.1f}s", flush=True)
    r = perplexity(model, windows, progress=True)
    rec = {"mode": mode, "ppl": r["ppl"], "nll_per_token": r["nll_per_token"],
           "n_tokens": r["n_tokens"], "n_windows": r["n_windows"], "seq_len": r["seq_len"],
           "n_layers": args.n_layers, "decoded_avg_bits": avg,
           "decoded_int_cost": cost,
           "bpp_avg_incl_scales": None if avg is None else avg + SCALE_BITS_PER_WEIGHT / db.group_size,
           "seconds": time.time() - t0}
    print(f"[evaluate] RESULT {mode} ppl={r['ppl']:.4f} nll/token={r['nll_per_token']:.6f} "
          f"n_tokens={r['n_tokens']}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[evaluate] wrote {args.out}", flush=True)


def _options(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-db", help="quantize all linears at every option")
    b.add_argument("--db-dir", default=str(DEFAULT_DB))
    b.add_argument("--options", type=_options, default=(2, 3, 4, 5, 6, 8))
    b.add_argument("--group-size", type=int, default=64)
    b.add_argument("--db-quantizer", choices=["gptq", "rtn"], default="gptq",
                   help="gptq: the deployment quantizer; rtn: fast experiments")
    b.add_argument("--damp", type=float, default=0.01)
    b.add_argument("--n-layers", type=int, default=N_LAYERS)
    b.add_argument("--calib-seqs", type=int, default=16)
    b.add_argument("--calib-len", type=int, default=2048)
    b.add_argument("--calib-seed", type=int, default=0)
    b.add_argument("--calib-split", default="train")
    b.set_defaults(fn=cmd_build_db)

    a = sub.add_parser("allocate", help="RCO allocation with the KL objective")
    a.add_argument("--db-dir", default=str(DEFAULT_DB))
    a.add_argument("--n-layers", type=int, default=N_LAYERS)
    a.add_argument("--avg-bits", type=float, default=3.0)
    a.add_argument("--steps", type=int, default=100)
    a.add_argument("--g", type=int, default=8)
    a.add_argument("--lr", type=float, default=0.1)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--tau0", type=float, default=1.0)
    a.add_argument("--tau-min", type=float, default=0.01)
    a.add_argument("--init", choices=["zero", "sensitivity"], default="zero")
    a.add_argument("--kl-seqs", type=int, default=16)
    a.add_argument("--kl-len", type=int, default=512)
    a.add_argument("--kl-seed", type=int, default=1)
    a.add_argument("--windows-per-sample", type=int, default=1)
    a.add_argument("--out", default=None)
    a.set_defaults(fn=cmd_allocate)

    e = sub.add_parser("evaluate", help="WikiText-2 test perplexity of one allocation")
    e.add_argument("--db-dir", default=str(DEFAULT_DB))
    e.add_argument("--n-layers", type=int, default=N_LAYERS)
    e.add_argument("--fp", action="store_true", help="the FP model, no allocation")
    e.add_argument("--uniform", type=int, default=None, help="every linear at this many bits")
    e.add_argument("--proxy-dp", action="store_true",
                   help="DP on the per-linear output-MSE sensitivities")
    e.add_argument("--avg-bits", type=float, default=3.0, help="budget for --proxy-dp")
    e.add_argument("--allocation", default=None, help="allocation JSON written by allocate")
    e.add_argument("--eval-len", type=int, default=2048)
    e.add_argument("--eval-windows", type=int, default=None, help="default: all (146)")
    e.add_argument("--out", default=None)
    e.set_defaults(fn=cmd_evaluate)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
