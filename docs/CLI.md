# gsq-rco command reference

`gsq-rco` is the one command line of this project. It has seven subcommands. Each has `--help`, which
prints the flags and examples. `python -m gsq_rco` is equivalent. Back to the [README](../README.md).

```bash
gsq-rco --help          # the seven commands
gsq-rco quantize --help # flags and examples for one command
```

Paths default to the repository, so commands work from any directory: the database is
`results/.rco_db`, the GSQ shards are `results/.gsq_cache`, the model is `models/Qwen2.5-0.5B/`, and the
dataset cache is `data/`. The model is not in the repository. Put the bf16 `Qwen/Qwen2.5-0.5B` snapshot
(`model.safetensors`, `config.json`, tokenizer files) in `models/Qwen2.5-0.5B/`.

| Command | What it does | Main output |
|---|---|---|
| `data` | Download and tokenize WikiText-2 and print row and token counts | JSON on stdout, optional `--out` |
| `eval` | WikiText-2 test perplexity of one model variant | JSON record (`--out`) |
| `build-db` | Quantize every linear at every option into the RCO database | `results/.rco_db/` |
| `allocate` | RCO bit allocation with the KL objective, or a baseline | allocation JSON (`--out`) |
| `quantize` | GSQ staged block-wise quantization, with optional export | shards, `run_log.json`, `mlx/` |
| `export` | Write an mlx-lm checkpoint from GSQ shards, plain RTN, or the database | checkpoint directory |
| `run` | `build-db`, `allocate`, `quantize`, `export`, `eval` in one resumable command | run directory |

Each command prints progress lines and a summary to stdout. `--out` writes the JSON record to a file.

## data

```bash
gsq-rco data                                   # token and window counts (downloads once)
gsq-rco data --seq-len 512 --out data_counts.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--seq-len N` | 2048 | Window length for the window count |
| `--out PATH` | none | Also write the JSON record here |

Output: `dataset`, `revision` (Salesforce/wikitext, pinned), `tokenizer` (Qwen2.5, no BOS), `seq_len`, and
per split (`train`, `validation`, `test`) the `rows`, `tokens`, and `windows_<seq_len>`.

## eval

```bash
gsq-rco eval --fp --out results/runs/eval_gptqdb_fp.json
gsq-rco eval --uniform 3 --db results/.rco_db --out results/runs/eval_gptqdb_u3.json
gsq-rco eval --proxy-dp --avg-bits 3.0 --db results/.rco_db --out results/runs/eval_gptqdb_proxy3.json
gsq-rco eval --allocation results/rco_allocation.json --db results/.rco_db --out results/runs/eval_gptqdb_rco3.json
gsq-rco eval --mlx-model results/runs/gsq_rco3/mlx --out results/runs/eval_gsq_rco3.json
```

Exactly one source is required:

| Flag | Meaning |
|---|---|
| `--fp` | The bf16 model, no allocation |
| `--uniform N` | Every linear at N bits, from the database (`--db`) |
| `--proxy-dp` | DP on the per-linear output-MSE sensitivities of the database, budget `--avg-bits` |
| `--allocation PATH` | An allocation JSON from `allocate`, quantized from the database |
| `--mlx-model DIR` | An exported mlx-lm checkpoint |

| Flag | Default | Meaning |
|---|---|---|
| `--db DIR` (alias `--db-dir`) | `results/.rco_db` | Database for `--uniform`, `--proxy-dp`, `--allocation` |
| `--n-layers N` | 24 | Layers of the model |
| `--avg-bits B` | 3.0 | Budget for `--proxy-dp` |
| `--eval-len N` | 2048 | Window length |
| `--max-windows N` (alias `--eval-windows`) | all (146) | Test windows to score |
| `--out PATH` | none | Write the JSON record |

Evaluation runs on the full WikiText-2 test split by default. Each window is scored with its 2,047
next-token targets. The JSON record has two shapes (section "Output formats").

## build-db

```bash
gsq-rco build-db --db results/.rco_db
```

Quantizes all 168 linears at every option. For each layer it runs the bf16 block on the calibration
windows, accumulates the input Hessians (position 0 dropped), quantizes each linear with the database
quantizer, and stores the codes, scales and sensitivity.

| Flag | Default | Meaning |
|---|---|---|
| `--db DIR` (alias `--db-dir`) | `results/.rco_db` | Output directory. Files are overwritten in place |
| `--n-layers N` | 24 | Layers of the model |
| `--options LIST` | `2,3,4,5,6,8` | Comma-separated bit widths |
| `--group-size N` | 64 | Quantization group size |
| `--db-quantizer {gptq,rtn}` | `gptq` | `gptq` is the deployment quantizer. `rtn` is for quick experiments |
| `--damp X` | 0.01 | GPTQ damping |
| `--calib-seqs N` | 16 | Calibration windows |
| `--calib-len N` | 2048 | Tokens per window |
| `--calib-seed N` | 0 | Window sampling seed |
| `--calib-split NAME` | `train` | Calibration split |

`build-db` always rebuilds. It does not check for an existing database. Use a new `--db` directory to keep
one, or use `run`, which reuses a database whose settings match (section "Resume and rebuild").

## allocate

```bash
gsq-rco allocate --db results/.rco_db --avg-bits 3.0 --out results/rco_allocation.json
gsq-rco allocate --db results/.rco_db --baseline proxy-dp --avg-bits 3.0 --out results/alloc_proxy3.json
```

Runs RCO (Alg. 1) with the App-B KL objective, or writes a baseline allocation with `--baseline`.

| Flag | Default | Meaning |
|---|---|---|
| `--db DIR` (alias `--db-dir`) | `results/.rco_db` | Database to allocate over |
| `--n-layers N` | 24 | Layers of the model |
| `--avg-bits B` | 3.0 | Average-bit budget. With `--baseline uniform`, the bit width |
| `--steps T` | 100 | RCO steps |
| `--g N` | 8 | Gumbel samples per step |
| `--lr X` | 0.1 | Adam step size on the logits |
| `--seed N` | 0 | Gumbel sampling seed |
| `--tau0 X` | 1.0 | Starting temperature (exponential schedule) |
| `--tau-min X` | 0.01 | Final temperature |
| `--init {zero,sensitivity}` | `zero` | `zero`: uniform start (paper default). `sensitivity`: logits `-S / mean(S)` from the database sensitivities |
| `--kl-seqs N` | 16 | Calibration windows for the KL objective |
| `--kl-len N` | 512 | Tokens per KL window |
| `--kl-seed N` | 1 | KL window sampling seed |
| `--windows-per-sample N` | 1 | KL windows scored per Gumbel sample |
| `--out PATH` | required | Allocation JSON to write |
| `--baseline {uniform,proxy-dp}` | none | Write a baseline allocation instead of running RCO |

## quantize

```bash
gsq-rco quantize --alloc results/rco_allocation.json --out results/runs/gsq_rco3
gsq-rco quantize --uniform-bits 3 --out results/runs/gsq_u3
gsq-rco quantize --alloc results/rco_allocation.json --resume --out results/runs/gsq_rco3
```

Quantizes the blocks in order. Each block gets a GPTQ warm start per linear, then trains the GSQ logits and
group scales in three stages (`qk`, `vo`, `mlp`), then hardens the codes. The next block is quantized from
the already-quantized prefix (`gsq/pipeline.py`, `gsq/staged.py`).

Exactly one of `--alloc` or `--uniform-bits` is required.

| Flag | Default | Meaning |
|---|---|---|
| `--alloc PATH` | none | Allocation JSON with `per_linear_bits` or `per_layer_bits` |
| `--uniform-bits {1..8}` | none | Every linear at this bit width (1 is ternary) |
| `--layers A-B` | `0-23` | Inclusive block range, for example `5-5` |
| `--n-cal N` | 128 | Calibration sequences |
| `--seq-len N` | 512 | Tokens per calibration sequence |
| `--cal-seed N` | 0 | Calibration sampling seed |
| `--steps N` | 200 | Lion steps per stage |
| `--lr-logits X` | 0.0001 | Logit learning rate (paper Table 7) |
| `--micro-batch N` | 2 | Calibration sequences per step |
| `--resamples N` | 2 | Gumbel resamples per step |
| `--damp X` | 0.01 | GPTQ damping |
| `--group-size N` | 64 | Quantization group size |
| `--seed N` | 0 | Training seed |
| `--scale-finetune-steps N` | 0 | App-H end-to-end scale steps. `0` is off |
| `--sanity-windows N` | 2 | Validation windows for the sanity perplexity. `0` skips it |
| `--out DIR` | `results/.gsq_cache` | Shards, `run_log.json`, and the default export |
| `--export-dir DIR` | `<out>/mlx` | mlx-lm export directory |
| `--no-export` | off | Skip the export |
| `--resume` | off | Reuse finished block shards in `--out` |

Group scales above 3 bits are frozen in the stage optimiser (`lion.scale_lr`, commit `c6f55b1`). 8-bit
linears keep their GPTQ codes and scales (commit `8df492a`). Shards record a fingerprint that includes this
rule, so shards from before the change are refused on `--resume`.

## export

```bash
gsq-rco export --out results/runs/gsq_u3/mlx                  # from the default shards
gsq-rco export --shards results/runs/gsq_u3 --out /path/to/mlx
gsq-rco export --rtn-bits 4 --out results/.rtn4               # plain RTN, no GSQ
gsq-rco export --gptq-db results/.rco_db --allocation results/rco_allocation.json --out results/.rco3
```

Writes an mlx-lm checkpoint. The source is one of three, and the default is the shards.

| Flag | Default | Meaning |
|---|---|---|
| `--out DIR` | required | Output checkpoint directory |
| `--base DIR` | `models/Qwen2.5-0.5B` | bf16 base checkpoint (norms, biases, embedding, config, tokenizer) |
| `--shards DIR` | `results/.gsq_cache` | GSQ shards. The default source |
| `--rtn-bits N` | none | Plain RTN at N bits |
| `--gptq-db DIR` | none | RCO database, with `--allocation` |
| `--allocation PATH` | none | Per-linear bits for `--gptq-db` |
| `--n-layers N` | 24 | Layers of the model |

## run

```bash
gsq-rco run --run-dir results/runs/rco3 --avg-bits 3.0 --dry-run   # print the plan only
gsq-rco run --run-dir results/runs/rco3 --avg-bits 3.0             # the full pipeline
gsq-rco run --run-dir results/runs/rco3 --avg-bits 3.0 --max-windows 8
```

Runs `build-db`, `allocate`, `quantize`, `export` and `eval` in that order, and records each step. Its
defaults are the defaults of the individual commands, with two names changed: `--rco-steps` is the RCO
step count (`allocate --steps`), and `--steps` is the Lion steps per stage (`quantize --steps`). `--dry-run`
prints the planned steps and their output paths, and writes nothing.

| Flag | Default | Meaning |
|---|---|---|
| `--run-dir DIR` | required | Run directory (see "Output formats") |
| `--dry-run` | off | Print the plan and run nothing |
| `--avg-bits B` | 3.0 | Average-bit budget |
| `--rco-steps T` | 100 | RCO steps |
| `--eval-len N` | 2048 | Evaluation window length |
| `--max-windows N` | all | Test windows to score |

Every other flag is one of the flags above with the same default: the database flags of `build-db`
(`--db`, `--options`, `--group-size`, `--db-quantizer`, `--damp`, `--calib-*`), the RCO flags of `allocate`
(`--g`, `--lr`, `--seed`, `--tau0`, `--tau-min`, `--init`, `--kl-*`, `--windows-per-sample`), and the GSQ
flags of `quantize` (`--layers`, `--n-cal`, `--seq-len`, `--cal-seed`, `--steps`, `--lr-logits`,
`--micro-batch`, `--resamples`, `--scale-finetune-steps`, `--sanity-windows`). Run `gsq-rco run --help`
for the full list.

## Output formats

### eval record

Database and bf16 modes (`--fp`, `--uniform`, `--proxy-dp`, `--allocation`):

```text
mode, ppl, nll_per_token, n_tokens, n_windows, seq_len, n_layers,
decoded_avg_bits, decoded_int_cost, bpp_avg_incl_scales, seconds
```

`decoded_avg_bits` is the integer cost over the total weights. `bpp_avg_incl_scales` adds 16/64 for the fp16
scales. Both are `null` for `--fp`.

MLX checkpoint mode (`--mlx-model`):

```text
ppl, nll_per_token, n_tokens, n_windows, seq_len, model, seconds, file_bytes, peak_mlx_gb
```

`file_bytes` is the sum of the `.safetensors` files in the checkpoint. `peak_mlx_gb` is the MLX peak in GiB.

### RCO database (`results/.rco_db/`)

- `manifest.json`: `format` (`rco_mlx.database/1`, kept for compatibility), `model_weights_sha256`, `options`,
  `group_size`, `quantizer`, `damp`, `signed_scale`, `n_layers`, `calibration` (split, seed, window count,
  window length, excluded positions `[0]`, token hash, Hessian rows), `hessian`, `counts`, `groups`
  (168 names, for example `0.q_proj`), and `sensitivity_file`. It is written last, atomically.
- `sensitivity.npy`: float64 `[168, K]`, the per-linear output MSE on the calibration Hessian.
- `L{layer:02d}.{linear}.b{bits}.codes.npy`: uint8 `[out, in]`.
- `L{layer:02d}.{linear}.b{bits}.scales.npy`: float32 `[out, in/group]`.

The database is about 2.1 GB on disk for 6 options. The build reads the model once, lazily, one block at a
time.

### allocation JSON

Keys: `method`, `config` (budget, steps, g, lr, seed, tau, init, KL settings), `database` (directory,
options, group size, quantizer, damp, identity), `budget` (`avg_bits_target`, `budget_int`, `total_units`,
`unit_params`), `expected_avg_bits` (continuous), `decoded_int_cost`, `decoded_avg_bits`,
`bpp_avg_incl_scales`, `checks` (`max_abs_budget_gap`, `budget_ok`, `decode_ok`), `per_linear_bits`
(`"<layer>.<linear>": bits`, read by `quantize --alloc` and `export --allocation`), `per_layer` (average bits
and parameters), `history` (one `{t, tau, cost, loss}` per step; `loss` is the mean over the g samples), and
`timing`.

### GSQ shards and `run_log.json`

Each block is `layer_XX.npz` with `meta` (JSON: layer, fingerprint, bits) and `stats` (JSON), and per linear
`<name>.codes` (uint8 `[out, in]`), `<name>.scales` (float16 `[out, in/group]`), and `<name>.bits`. The
fingerprint lists the settings that must match on `--resume`.

`run_log.json` has `config`, `layers` (per block: `stages` with `objective_init`, `objective_final`, steps,
time; `linears` with `bits`, `mse_gptq_init`, `mse_hard_final`, `flip_frac`; `peak_mlx_bytes`; `seconds`),
`scale_finetune`, `sanity`, `export` (directory, `bits_per_weight` with `code_bpp`, `native_bpp`, `mlx_bpp`,
`shard_bpp`), `seconds_total`, and `max_rss_bytes`.

### MLX export directory

`model.safetensors` (626 tensors: packed uint32 codes with bf16 scales and biases per group, and the bf16
tied embedding), `config.json`, and the tokenizer files. It loads with `mlx_lm.load`.

### run directory

```text
steps/<name>.json   record of each finished step: config, chained hash, outputs, result, seconds
allocation.json     RCO allocation
gsq/                GSQ shards and run_log.json
mlx/                exported checkpoint
eval.json           evaluation record
run.json            summary of the last invocation
```

## Resume and rebuild

- **`quantize --resume`** reuses the finished `layer_XX.npz` shards in `--out` and replays them into the
  quantized prefix. Finished blocks are never retrained. Shards whose fingerprint or bit widths differ from
  the current arguments are refused with an error. Delete them or change the arguments.
- **`build-db`** always rebuilds into `--db`. Its manifest is written last and atomically, so an interrupted
  build leaves no manifest behind.
- **`run`** chains a hash through the steps. Each step's hash covers its own settings and the previous
  step's hash. A step is skipped when its record has the same hash and its outputs exist. Changing an early
  setting therefore re-runs every later step. If the database already exists and its settings match, the
  `build-db` step reuses it and does not rebuild (`status: reused`). If the settings differ, `run` stops
  with an error and names the differing keys. A step with no record is re-executed, so the first `run` in a
  new run directory calls the `build-db` step, which reuses an existing matching database.

## Memory and time

Measured on the 16 GiB M3 for the committed settings (sections 2 and 9 of [RESULTS.md](RESULTS.md)):

| Command | Time | Peak memory |
|---|---:|---|
| `data` (cached) | 0.13 s | small |
| `build-db` (6 options) | 556.7 s | footprint 1.37 GiB |
| `allocate` (100 steps, g = 8) | 2,825.1 s | footprint 2.33 GiB |
| `quantize` (GSQ, 24 blocks) | 5,558 to 5,825 s | max RSS 2.5 to 2.9 GiB, peak MLX 3.12 GiB |
| `eval` (GPTQ database) | 123 to 140 s | footprint 2.33 to 2.38 GiB (bf16: 1.76 GiB) |
| `eval` (MLX checkpoint) | 125 to 127 s | footprint 1.61 GiB |

- Run one heavy command at a time. Do not run two model jobs in parallel.
- `quantize` is the heaviest step. Its peak is set by the calibration activations and the stage
  optimiser. The flags `--n-cal`, `--seq-len`, `--micro-batch` and `--resamples` scale that work. Their memory
  effect was not measured, so treat them as untested knobs.
- `--layers A-B` quantizes a block range. It is useful for a quick check (for example `--layers 5-5 --n-cal 16
  --steps 50`) before a full run.
- The timings are from the run logs. The data download is not timed in any log.
