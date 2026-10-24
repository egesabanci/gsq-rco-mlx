"""`gsq-rco run`: build-db -> allocate -> quantize -> export -> eval, resumable by config hash.

Run directory layout:
  steps/<name>.json   record of each finished step: config, hash, outputs, result, seconds
  allocation.json     RCO allocation (allocate)
  gsq/                GSQ block shards and run_log.json (quantize)
  mlx/                exported mlx-lm checkpoint (export)
  eval.json           WikiText-2 perplexity of mlx/ (eval)
  run.json            summary of the last invocation

The config hash of a step chains its own settings to the hash of the step before it, so changing an
early setting re-runs every later step. A step is skipped when its record exists, the hash matches
and its outputs are on disk. build-db is never rebuilt over an existing database: an existing manifest
is reused when its spec matches the run's (the step is recorded as "reused"), otherwise the run stops.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

from gsq_rco.common.data import MODEL_DIR

STEP_NAMES = ("build-db", "allocate", "quantize", "export", "eval")


def config_hash(config: dict, previous: str | None) -> str:
    """sha256 of the canonical JSON of (this step's config, the hash of the previous step)."""
    payload = json.dumps({"config": config, "previous": previous}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_plan(args) -> list[dict]:
    """The five steps in order, each with its config, chained hash and output paths."""
    run = Path(args.run_dir)
    steps, previous = [], None

    def add(name: str, config: dict, outputs: dict) -> None:
        nonlocal previous
        previous = config_hash(config, previous)
        steps.append(dict(name=name, config=config, hash=previous,
                          outputs={k: str(v) for k, v in outputs.items()}))

    add("build-db", dict(options=list(args.options), group_size=args.group_size,
                         db_quantizer=args.db_quantizer, damp=args.damp, n_layers=args.n_layers,
                         calib_seqs=args.calib_seqs, calib_len=args.calib_len,
                         calib_seed=args.calib_seed, calib_split=args.calib_split),
        {"manifest": Path(args.db) / "manifest.json"})
    add("allocate", dict(avg_bits=args.avg_bits, rco_steps=args.rco_steps, g=args.g, lr=args.lr,
                         seed=args.seed, tau0=args.tau0, tau_min=args.tau_min, init=args.init,
                         kl_seqs=args.kl_seqs, kl_len=args.kl_len, kl_seed=args.kl_seed,
                         windows_per_sample=args.windows_per_sample, n_layers=args.n_layers),
        {"allocation": run / "allocation.json"})
    add("quantize", dict(layers=args.layers, n_cal=args.n_cal, seq_len=args.seq_len,
                         cal_seed=args.cal_seed, steps=args.steps, lr_logits=args.lr_logits,
                         micro_batch=args.micro_batch, resamples=args.resamples, damp=args.damp,
                         group_size=args.group_size, seed=args.seed,
                         scale_finetune_steps=args.scale_finetune_steps),
        {"run_log": run / "gsq" / "run_log.json"})
    add("export", dict(group_size=args.group_size),
        {"safetensors": run / "mlx" / "model.safetensors"})
    add("eval", dict(eval_len=args.eval_len, max_windows=args.eval_windows),
        {"record": run / "eval.json"})
    return steps


def _record_path(run: Path, name: str) -> Path:
    return run / "steps" / f"{name}.json"


def action_for(step: dict, run: Path) -> str:
    """'skip' when the step's record carries the same hash and every output is on disk, else 'run'."""
    record = _record_path(run, step["name"])
    outputs_ok = all(Path(p).exists() for p in step["outputs"].values())
    if outputs_ok and record.exists() and json.loads(record.read_text()).get("hash") == step["hash"]:
        return "skip"
    return "run"


def _check_database_spec(manifest_path: Path, config: dict) -> None:
    m = json.loads(manifest_path.read_text())
    c = m["calibration"]
    have = {"options": list(m["options"]), "group_size": m["group_size"], "quantizer": m["quantizer"],
            "damp": m["damp"], "n_layers": m["n_layers"], "calib_seqs": c["n_windows"],
            "calib_len": c["window_len"], "calib_seed": c["seed"], "calib_split": c["split"]}
    want = {k: (list(config[k]) if k == "options" else config[k]) for k in
            ("options", "group_size", "damp", "n_layers", "calib_seqs", "calib_len", "calib_seed",
             "calib_split")}
    want["quantizer"] = config["db_quantizer"]
    diffs = sorted(k for k in want if have[k] != want[k])
    if diffs:
        raise SystemExit(f"{manifest_path} was built with other settings ({', '.join(diffs)}); "
                         "point --db at another directory or delete it")


def _run_build_db(args, step: dict) -> dict:
    manifest = Path(step["outputs"]["manifest"])
    if manifest.exists():
        _check_database_spec(manifest, step["config"])
        return {"status": "reused", "manifest": str(manifest)}
    from gsq_rco.rco.driver import build_database_cmd
    config = dict(step["config"], options=tuple(step["config"]["options"]))
    build_database_cmd(SimpleNamespace(db=args.db, **config))
    return {"status": "built", "manifest": str(manifest)}


def _run_allocate(args, step: dict) -> dict:
    from gsq_rco.rco.driver import cmd_allocate
    cmd_allocate(SimpleNamespace(
        db=args.db, n_layers=args.n_layers, avg_bits=args.avg_bits, steps=args.rco_steps,
        g=args.g, lr=args.lr, seed=args.seed, tau0=args.tau0, tau_min=args.tau_min, init=args.init,
        kl_seqs=args.kl_seqs, kl_len=args.kl_len, kl_seed=args.kl_seed,
        windows_per_sample=args.windows_per_sample, out=step["outputs"]["allocation"],
        baseline=None))
    return {"allocation": step["outputs"]["allocation"]}


def _run_quantize(args, step: dict) -> dict:
    from gsq_rco.gsq.pipeline import run_quantize
    out = Path(step["outputs"]["run_log"]).parent
    resume = any(out.glob("layer_*.npz"))  # finished blocks are replayed, never retrained
    run_quantize(SimpleNamespace(
        alloc=str(Path(args.run_dir) / "allocation.json"), uniform_bits=None, layers=args.layers,
        n_cal=args.n_cal, seq_len=args.seq_len, cal_seed=args.cal_seed, steps=args.steps,
        lr_logits=args.lr_logits, micro_batch=args.micro_batch, resamples=args.resamples,
        damp=args.damp, group_size=args.group_size, seed=args.seed,
        sanity_windows=args.sanity_windows, scale_finetune_steps=args.scale_finetune_steps,
        out=str(out), export_dir=None, no_export=True, resume=resume))
    return {"shards": str(out), "resumed": resume}


def _run_export(args, step: dict) -> dict:
    from gsq_rco.gsq.export import export_checkpoint, load_shards
    quantized = load_shards(str(Path(args.run_dir) / "gsq"))
    return export_checkpoint(quantized, Path(step["outputs"]["safetensors"]).parent, MODEL_DIR)


def _run_eval(args, step: dict) -> dict:
    from gsq_rco.common.evaluate import evaluate_mlx_model
    rec = evaluate_mlx_model(Path(args.run_dir) / "mlx", eval_len=args.eval_len,
                             max_windows=args.eval_windows)
    Path(step["outputs"]["record"]).write_text(json.dumps(rec, indent=1))
    return {"ppl": rec["ppl"], "nll_per_token": rec["nll_per_token"], "n_tokens": rec["n_tokens"]}


RUNNERS = {"build-db": _run_build_db, "allocate": _run_allocate, "quantize": _run_quantize,
           "export": _run_export, "eval": _run_eval}


def run_pipeline(args, runners: dict | None = None) -> dict:
    """Run (or resume) the five steps. runners maps step names to callables(args, step) and is
    injectable for tests; the default runners are the functions behind the individual commands."""
    run = Path(args.run_dir)
    steps = build_plan(args)
    plan = [dict(name=s["name"], action=action_for(s, run), hash=s["hash"], outputs=s["outputs"])
            for s in steps]
    if args.dry_run:
        out = {"dry_run": True, "run_dir": str(run), "plan": plan}
        print(json.dumps(out, indent=1))
        return out
    (run / "steps").mkdir(parents=True, exist_ok=True)
    runners = RUNNERS if runners is None else runners
    results = {}
    for step, item in zip(steps, plan):
        if item["action"] == "skip":
            print(f"[run] {step['name']}: skipped (hash {step['hash'][:12]}, outputs present)", flush=True)
            results[step["name"]] = {"status": "skipped"}
            continue
        print(f"[run] {step['name']}: running (hash {step['hash'][:12]})", flush=True)
        t0 = time.time()
        result = runners[step["name"]](args, step)
        record = dict(name=step["name"], hash=step["hash"], config=step["config"],
                      outputs=step["outputs"], result=result,
                      seconds=round(time.time() - t0, 2))
        _record_path(run, step["name"]).write_text(json.dumps(record, indent=1))
        results[step["name"]] = result
    summary = {"run_dir": str(run), "plan": plan, "results": results}
    (run / "run.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return summary
