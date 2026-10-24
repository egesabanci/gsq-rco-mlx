"""Commands behind `gsq-rco build-db`, `gsq-rco allocate` and `gsq-rco eval` (WP6, paper App B, Alg.1).

  build_database_cmd  quantize every linear at every option with the deployment quantizer (GPTQ with
                      signed scales by default); store codes, scales and per-linear output-MSE
                      sensitivities. Hessians come from WikiText-2 train windows of the FP model.
  cmd_allocate        RCO (Alg.1) with the KL objective (Eq.6) over the database, writing the allocation
                      JSON. --baseline uniform|proxy-dp writes a baseline in the same JSON schema.
  cmd_evaluate        WikiText-2 test perplexity of the FP model, of a uniform or proxy-DP allocation, of
                      an allocation JSON, or of an exported mlx-lm checkpoint.

Run the memory-heavy commands one at a time (the shared memory lock, RULES item 4).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gsq_rco.common.data import MODEL_DIR, REPO
from gsq_rco.common.model import N_LAYERS, linear_path

DEFAULT_DB = REPO / "results" / ".rco_db"
DEFAULT_OUT = REPO / "results" / "rco_allocation.json"
BUDGET_TOL = 1e-4           # |C - B| per step, the acceptance bound of the brief
SCALE_BITS_PER_WEIGHT = 16  # one fp16 scale per group; bpp = code bits + 16 / group_size


def model_fingerprint() -> str:
    from gsq_rco.rco.database import sha256_files
    return sha256_files([MODEL_DIR / "config.json", MODEL_DIR / "model.safetensors"])


def calibration_tokens(seqs: int, length: int, seed: int, split: str) -> np.ndarray:
    from gsq_rco.common.data import calibration_windows
    return calibration_windows(seqs, length, seed=seed, split=split)


def load_model(n_layers: int, lazy: bool = False):
    """The FP Qwen2.5-0.5B cut to n_layers blocks. lazy=True (build-db only) leaves every weight
    unloaded until its block is used, through mlx_lm load_model(lazy=True); the other commands
    need all their blocks resident and use common.model.load_fp_model."""
    if lazy:
        from mlx_lm.utils import load_model as mlx_load_model
        model, _ = mlx_load_model(MODEL_DIR, lazy=True)
        return model
    from gsq_rco.common.model import load_fp_model
    from gsq_rco.rco.allocate import truncate_layers
    return truncate_layers(load_fp_model(), n_layers)


def _open_database(db_dir, n_layers: int, model_sha: str):
    from gsq_rco.rco.database import Database
    db = Database(db_dir)
    c = db.manifest["calibration"]
    db.verify(model_sha, calibration_tokens(c["n_windows"], c["window_len"], c["seed"], c["split"]))
    if n_layers != db.n_layers:
        raise SystemExit(f"--n-layers {n_layers} but the database has {db.n_layers} layers")
    return db


def _database_record(db, db_dir) -> dict:
    return {"dir": str(Path(db_dir).resolve()), "options": list(db.options),
            "group_size": db.group_size, "quantizer": db.manifest["quantizer"],
            "damp": db.manifest["damp"], "n_layers": db.n_layers,
            "identity": {k: db.manifest[k] for k in
                         ("format", "model_weights_sha256", "calibration", "hessian")}}


def _check_allocation_identity(saved: dict, db, path) -> None:
    identity = saved["database"]["identity"]
    if identity != {k: db.manifest[k] for k in identity}:
        raise SystemExit(f"{path} was made from a different database")


def build_database_cmd(args) -> None:
    """`gsq-rco build-db`: write the RCO quantized database (manifest.json plus per-group files)."""
    from gsq_rco.rco.database import DBSpec, build_database
    t0 = time.time()
    spec = DBSpec(options=args.options, group_size=args.group_size, quantizer=args.db_quantizer,
                  damp=args.damp, n_layers=args.n_layers, calib_split=args.calib_split,
                  calib_seed=args.calib_seed)
    tokens = calibration_tokens(args.calib_seqs, args.calib_len, args.calib_seed, args.calib_split)
    print(f"[build-db] quantizer={spec.quantizer} options={spec.options} "
          f"group={spec.group_size} damp={spec.damp} layers={spec.n_layers} "
          f"calibration={tokens.shape[0]}x{tokens.shape[1]} "
          f"({tokens.shape[0] * (tokens.shape[1] - 1)} Hessian rows) -> {args.db}", flush=True)
    model = load_model(args.n_layers, lazy=True)
    print(f"[build-db] model opened lazily in {time.time() - t0:.1f}s", flush=True)
    manifest = build_database(model, tokens, spec, args.db, model_fingerprint(), log=print,
                              release=True)
    print(f"[build-db] wrote {Path(args.db) / 'manifest.json'} "
          f"({len(manifest['groups'])} linears) in {time.time() - t0:.1f}s", flush=True)


def cmd_allocate(args) -> None:
    """`gsq-rco allocate`: RCO with the KL objective, or a baseline allocation (--baseline)."""
    from gsq_rco.common.evaluate import CHUNK
    from gsq_rco.rco.allocate import KLObjective, allocation_summary, rco_allocate
    from gsq_rco.rco.loop import integer_weights
    if args.out is None:
        raise SystemExit("--out is required")
    out = Path(args.out).resolve()
    if out == DEFAULT_OUT.resolve() and args.n_layers != N_LAYERS:
        raise SystemExit("results/rco_allocation.json is written only by the full 24-layer run; "
                         "pass another --out for partial runs")
    if getattr(args, "baseline", None) is not None:
        write_baseline(args, out)
        return
    t0 = time.time()
    sha = model_fingerprint()
    db = _open_database(args.db, args.n_layers, sha)
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
        "database": _database_record(db, args.db),
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


def write_baseline(args, out: Path) -> None:
    """Baseline allocation in the allocate JSON schema, so `eval --allocation` can score it.

    uniform: every linear at --avg-bits (an integer option). proxy-dp: DP on the per-linear
    output-MSE sensitivities at --avg-bits (the same budget grid as RCO)."""
    from gsq_rco.rco.allocate import (allocation_summary, budget_grid, decoded_cost,
                                      proxy_dp_choice, uniform_choice)
    from gsq_rco.rco.loop import integer_weights
    t0 = time.time()
    db = _open_database(args.db, args.n_layers, model_fingerprint())
    total = int(integer_weights(db.counts)[0].sum())
    if args.baseline == "uniform":
        if float(args.avg_bits) != int(args.avg_bits):
            raise SystemExit("--baseline uniform needs an integer --avg-bits")
        choice = uniform_choice(db, int(args.avg_bits))
        method = f"uniform {int(args.avg_bits)}-bit baseline"
        budget_int = None
    else:
        _, _, budget_int = budget_grid(args.avg_bits, total, len(db.groups))
        choice = proxy_dp_choice(db, args.avg_bits)
        method = "proxy-DP baseline (DP on per-linear output-MSE sensitivities, same database)"
    cost, avg = decoded_cost(db, choice)
    if budget_int is None:
        budget_int = cost
    per_linear, per_layer = allocation_summary(db, choice)
    rec = {
        "method": method,
        "config": {"baseline": args.baseline, "avg_bits": args.avg_bits, "n_layers": args.n_layers},
        "database": _database_record(db, args.db),
        "budget": {"avg_bits_target": args.avg_bits, "budget_int": budget_int,
                   "total_units": total},
        "decoded_int_cost": cost,
        "decoded_avg_bits": avg,
        "bpp_avg_incl_scales": avg + SCALE_BITS_PER_WEIGHT / db.group_size,
        "checks": {"decode_ok": bool(cost <= budget_int)},
        "per_linear_bits": per_linear,
        "per_layer": per_layer,
        "history": [],
        "timing": {"total_s": time.time() - t0},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"[allocate] {method}: decoded {cost}/{budget_int} -> {avg:.6f} code bits; "
          f"wrote {out}", flush=True)


def quantized_from_database(db_dir, allocation: str, n_layers: int = N_LAYERS) -> dict:
    """{module path: dict(codes, scales, bits, group_size)} with each linear at the bits that the
    allocation JSON gives it, read from the GPTQ database (for `gsq-rco export --gptq-db`)."""
    db = _open_database(db_dir, n_layers, model_fingerprint())
    with open(allocation) as f:
        saved = json.load(f)
    _check_allocation_identity(saved, db, allocation)
    bits = saved["per_linear_bits"]
    quantized = {}
    for i, (L, name) in enumerate(db.groups):
        b = int(bits[f"{L}.{name}"])
        codes, scales = db.codes_scales(i, db.options.index(b))
        quantized[linear_path(L, name)] = dict(codes=codes, scales=scales, bits=b,
                                               group_size=db.group_size)
    return quantized


def cmd_evaluate(args) -> None:
    """`gsq-rco eval`: exactly one of --fp, --uniform, --proxy-dp, --allocation, --mlx-model."""
    chosen = [args.fp, args.uniform is not None, args.proxy_dp, args.allocation is not None,
              args.mlx_model is not None]
    if sum(chosen) != 1:
        raise SystemExit("choose exactly one of --fp, --uniform, --proxy-dp, --allocation, --mlx-model")
    if args.mlx_model is not None:
        from gsq_rco.common.evaluate import evaluate_mlx_model
        rec = evaluate_mlx_model(args.mlx_model, eval_len=args.eval_len,
                                 max_windows=args.eval_windows)
        print(f"[evaluate] RESULT mlx-model ppl={rec['ppl']:.4f} nll/token={rec['nll_per_token']:.6f} "
              f"n_tokens={rec['n_tokens']} file={rec['file_bytes'] / 1e6:.1f} MB "
              f"peak MLX {rec['peak_mlx_gb']:.2f} GB", flush=True)
        _write_record(rec, args.out)
        return
    from gsq_rco.common.data import eval_windows
    from gsq_rco.common.evaluate import perplexity
    from gsq_rco.rco.allocate import decoded_cost, install_choice, proxy_dp_choice, uniform_choice
    t0 = time.time()
    choice = None
    db = None
    if not args.fp:
        db = _open_database(args.db, args.n_layers, model_fingerprint())
    if args.uniform is not None:
        mode, choice = f"uniform-{args.uniform}", uniform_choice(db, args.uniform)
    elif args.proxy_dp:
        mode, choice = f"proxy-dp-{args.avg_bits}", proxy_dp_choice(db, args.avg_bits)
    elif args.allocation is not None:
        with open(args.allocation) as f:
            saved = json.load(f)
        _check_allocation_identity(saved, db, args.allocation)
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
    _write_record(rec, args.out)


def _write_record(rec: dict, out) -> None:
    if out:
        with open(out, "w") as f:
            json.dump(rec, f, indent=1)
        print(f"[evaluate] wrote {out}", flush=True)
