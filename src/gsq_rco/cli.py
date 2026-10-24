"""gsq-rco: one command line for the GSQ and RCO quantization of Qwen2.5-0.5B.

Every command prints a log and, where it produces one, a JSON record. Commands that load the model
are memory heavy: run them one at a time (RULES item 4 lock when several agents share the machine).
Run `gsq-rco COMMAND --help` for the options and examples of one command.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gsq_rco.common.data import MODEL_DIR, REPO

DEFAULT_DB = REPO / "results" / ".rco_db"
DEFAULT_SHARDS = REPO / "results" / ".gsq_cache"
N_LAYERS = 24

EXAMPLES = {
    "data": """examples:
  gsq-rco data                      # token and window counts of WikiText-2 (downloads once)
  gsq-rco data --seq-len 512 --out results/data_counts.json""",
    "eval": """examples:
  gsq-rco eval --fp --out results/runs/eval_fp.json
  gsq-rco eval --uniform 3 --db results/.rco_db --out results/runs/eval_u3.json
  gsq-rco eval --allocation results/rco_allocation.json --db results/.rco_db
  gsq-rco eval --proxy-dp --avg-bits 3.0 --db results/.rco_db
  gsq-rco eval --mlx-model results/.gsq_cache/mlx --max-windows 8""",
    "build-db": """examples:
  gsq-rco build-db --db results/.rco_db                 # GPTQ database, 6 options, 16 x 2048 windows
  gsq-rco build-db --db /tmp/db --db-quantizer rtn --options 2,3 --n-layers 2   # quick check""",
    "allocate": """examples:
  gsq-rco allocate --db results/.rco_db --avg-bits 3.0 --out results/alloc_rco3.json
  gsq-rco allocate --db results/.rco_db --baseline proxy-dp --avg-bits 3.0 --out results/alloc_proxy3.json
  gsq-rco allocate --db results/.rco_db --baseline uniform --avg-bits 3 --out results/alloc_u3.json""",
    "quantize": """examples:
  gsq-rco quantize --alloc results/rco_allocation.json --out results/.gsq_cache
  gsq-rco quantize --uniform-bits 3 --layers 5-5 --n-cal 16 --seq-len 512 --steps 50 --out /tmp/q5
  gsq-rco quantize --alloc alloc.json --resume --out results/.gsq_cache   # reuse finished blocks""",
    "export": """examples:
  gsq-rco export --out results/.gsq_cache/mlx                          # from GSQ shards (default dir)
  gsq-rco export --shards /tmp/q5 --out /tmp/q5/mlx
  gsq-rco export --rtn-bits 4 --out results/.rtn4
  gsq-rco export --gptq-db results/.rco_db --allocation results/rco_allocation.json --out results/.rco3""",
    "run": """examples:
  gsq-rco run --run-dir results/.runs/rco3 --avg-bits 3.0 --dry-run    # print the plan only
  gsq-rco run --run-dir results/.runs/rco3 --avg-bits 3.0              # build-db -> allocate -> quantize -> export -> eval
  gsq-rco run --run-dir results/.runs/rco3 --avg-bits 3.0 --max-windows 8   # rerun; finished steps are skipped""",
}


def _options(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.split(","))


def _command(sub, name: str, help_text: str):
    return sub.add_parser(name, help=help_text, description=help_text,
                          epilog=EXAMPLES[name], formatter_class=argparse.RawDescriptionHelpFormatter)


def _add_db(p) -> None:
    p.add_argument("--db", "--db-dir", dest="db", default=str(DEFAULT_DB),
                   help="RCO quantized database directory (default results/.rco_db)")
    p.add_argument("--n-layers", type=int, default=N_LAYERS)


def _add_build(p) -> None:
    p.add_argument("--options", type=_options, default=(2, 3, 4, 5, 6, 8),
                   help="comma-separated bit-widths of the database")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--db-quantizer", choices=["gptq", "rtn"], default="gptq",
                   help="gptq: the deployment quantizer; rtn: fast experiments")
    p.add_argument("--damp", type=float, default=0.01, help="GPTQ damping")
    p.add_argument("--calib-seqs", type=int, default=16)
    p.add_argument("--calib-len", type=int, default=2048)
    p.add_argument("--calib-seed", type=int, default=0)
    p.add_argument("--calib-split", default="train")


def _add_allocate(p, steps_flag: str = "--steps", steps_dest: str = "steps") -> None:
    p.add_argument("--avg-bits", type=float, default=3.0, help="average-bit budget (RCO), "
                   "or the bit-width of --baseline uniform / the budget of --baseline proxy-dp")
    p.add_argument(steps_flag, dest=steps_dest, type=int, default=100, help="RCO steps T")
    p.add_argument("--g", type=int, default=8, help="Gumbel samples per step")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tau0", type=float, default=1.0)
    p.add_argument("--tau-min", type=float, default=0.01)
    p.add_argument("--init", choices=["zero", "sensitivity"], default="zero")
    p.add_argument("--kl-seqs", type=int, default=16)
    p.add_argument("--kl-len", type=int, default=512)
    p.add_argument("--kl-seed", type=int, default=1)
    p.add_argument("--windows-per-sample", type=int, default=1)


def _add_quantize(p, skip=()) -> None:
    """GSQ options. `skip` names options another group of the same parser already defines (run)."""
    p.add_argument("--layers", default="0-23", help="block range to quantize, e.g. 0-23 or 5-5")
    p.add_argument("--n-cal", type=int, default=128, help="calibration sequences")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--cal-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200, help="Lion steps per stage")
    p.add_argument("--lr-logits", type=float, default=1e-4, help="logit lr (Table 7)")
    p.add_argument("--micro-batch", type=int, default=2)
    p.add_argument("--resamples", type=int, default=2, help="Gumbel resamples per step")
    if "damp" not in skip:
        p.add_argument("--damp", type=float, default=0.01, help="GPTQ damping")
    if "group_size" not in skip:
        p.add_argument("--group-size", type=int, default=64)
    if "seed" not in skip:
        p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale-finetune-steps", type=int, default=0, help="App H steps (0 = off)")
    p.add_argument("--sanity-windows", type=int, default=2,
                   help="validation windows for the sanity perplexity (0 to skip)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gsq-rco",
        description="GSQ (Gumbel-Softmax quantization) and RCO (bit allocation) for Qwen2.5-0.5B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands: data, build-db, allocate, quantize, export, eval, run. "
               "Each has --help with examples.")
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p = _command(sub, "data", "download and tokenize WikiText-2; print row and token counts")
    p.add_argument("--seq-len", type=int, default=2048, help="window length for the window count")
    p.add_argument("--out", default=None, help="also write the JSON record here")
    p.set_defaults(fn=cmd_data)

    p = _command(sub, "eval", "WikiText-2 test perplexity of a model variant (JSON record)")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fp", action="store_true", help="the FP model, no allocation")
    mode.add_argument("--uniform", type=int, help="every linear at this many bits (from --db)")
    mode.add_argument("--proxy-dp", action="store_true",
                      help="DP on the per-linear output-MSE sensitivities (budget --avg-bits)")
    mode.add_argument("--allocation", help="allocation JSON written by allocate (uses --db)")
    mode.add_argument("--mlx-model", help="exported mlx-lm checkpoint directory")
    _add_db(p)
    p.add_argument("--avg-bits", type=float, default=3.0, help="budget for --proxy-dp")
    p.add_argument("--eval-len", type=int, default=2048)
    p.add_argument("--max-windows", "--eval-windows", dest="eval_windows", type=int, default=None,
                   help="test windows to score (default: all 146)")
    p.add_argument("--out", default=None, help="write the JSON record here")
    p.set_defaults(fn=cmd_eval)

    p = _command(sub, "build-db", "quantize every linear at every option into the RCO database")
    _add_db(p)
    _add_build(p)
    p.set_defaults(fn=cmd_build_db)

    p = _command(sub, "allocate", "RCO bit allocation with the KL objective (or a baseline)")
    _add_db(p)
    _add_allocate(p)
    p.add_argument("--out", default=None, required=True, help="allocation JSON to write")
    p.add_argument("--baseline", choices=["uniform", "proxy-dp"], default=None,
                   help="write a baseline allocation instead of running RCO")
    p.set_defaults(fn=cmd_allocate)

    p = _command(sub, "quantize", "GSQ staged block-wise quantization (shards, run_log, export)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--alloc", help="allocation JSON (per_linear_bits or per_layer_bits)")
    src.add_argument("--uniform-bits", type=int, choices=range(1, 9))
    _add_quantize(p)
    p.add_argument("--out", default=str(DEFAULT_SHARDS), help="shard, log and export directory")
    p.add_argument("--export-dir", help="mlx-lm export directory (default <out>/mlx)")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--resume", action="store_true", help="reuse the finished block shards in --out")
    p.set_defaults(fn=cmd_quantize)

    p = _command(sub, "export", "write an mlx-lm checkpoint from GSQ shards, RTN or the database")
    p.add_argument("--out", required=True, help="output directory for the mlx-lm checkpoint")
    p.add_argument("--base", default=str(MODEL_DIR), help="bf16 base checkpoint directory")
    origin = p.add_mutually_exclusive_group()
    origin.add_argument("--shards", default=None,
                        help="directory of GSQ layer_XX.npz shards (default results/.gsq_cache)")
    origin.add_argument("--rtn-bits", type=int, help="plain RTN at this bit-width")
    origin.add_argument("--gptq-db", help="RCO database directory (with --allocation)")
    p.add_argument("--allocation", help="allocation JSON giving each linear its bits (--gptq-db)")
    p.add_argument("--n-layers", type=int, default=N_LAYERS)
    p.set_defaults(fn=cmd_export)

    p = _command(sub, "run", "build-db -> allocate -> quantize -> export -> eval, resumable")
    p.add_argument("--run-dir", required=True,
                   help="run directory: steps/ records, allocation.json, gsq/, mlx/, eval.json")
    _add_db(p)
    _add_build(p)
    _add_allocate(p, steps_flag="--rco-steps", steps_dest="rco_steps")  # --seed shared with GSQ
    _add_quantize(p, skip={"damp", "group_size", "seed"})  # --damp, --group-size shared with build-db
    p.add_argument("--eval-len", type=int, default=2048)
    p.add_argument("--max-windows", dest="eval_windows", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    p.set_defaults(fn=cmd_run)
    return ap


# ------------------------------------------------------------------ commands ---

def _emit(record: dict, out) -> None:
    text = json.dumps(record, indent=1)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")


def cmd_data(args) -> int:
    from gsq_rco.common.data import DATASET, DATASET_REVISION, SPLITS, split_rows, tokenize_split
    splits = {}
    for split in SPLITS:
        ids = tokenize_split(split)
        splits[split] = {"rows": split_rows(split), "tokens": int(ids.size),
                         f"windows_{args.seq_len}": int(ids.size // args.seq_len)}
    _emit({"dataset": DATASET, "revision": DATASET_REVISION, "tokenizer": "Qwen2.5 (no BOS)",
           "seq_len": args.seq_len, "splits": splits}, args.out)
    return 0


def cmd_eval(args) -> int:
    from gsq_rco.rco.driver import cmd_evaluate
    cmd_evaluate(args)
    return 0


def cmd_build_db(args) -> int:
    from gsq_rco.rco.driver import build_database_cmd
    build_database_cmd(args)
    return 0


def cmd_allocate(args) -> int:
    from gsq_rco.rco.driver import cmd_allocate as allocate
    allocate(args)
    return 0


def cmd_quantize(args) -> int:
    from gsq_rco.gsq.pipeline import run_quantize
    run_quantize(args)
    return 0


def cmd_export(args) -> int:
    from gsq_rco.gsq.export import export_checkpoint, load_shards, rtn_quantized
    if args.gptq_db is not None:
        if args.allocation is None:
            raise SystemExit("--gptq-db needs --allocation")
        from gsq_rco.rco.driver import quantized_from_database
        quantized = quantized_from_database(args.gptq_db, args.allocation, args.n_layers)
    elif args.rtn_bits is not None:
        quantized = rtn_quantized(args.base, args.rtn_bits)
    else:
        shards = args.shards or str(DEFAULT_SHARDS)
        quantized = load_shards(shards)
    if not quantized:
        raise SystemExit("no quantized linears found")
    summary = export_checkpoint(quantized, args.out, args.base)
    _emit(summary, None)
    return 0


def cmd_run(args) -> int:
    from gsq_rco.run import run_pipeline
    run_pipeline(args)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
