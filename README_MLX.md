# GSQ + RCO for MLX (Apple Silicon) — scaffold

Target: port paper algorithms to `mlx.core` so we can quantize **very small models**
entirely on Mac (M3), then export to `mlx-lm` / GGUF for inference.

Papers in this folder: `gsq.pdf` (scalar quant via Gumbel-Softmax),
`rco.pdf` (budget allocation via Riemannian manifold).

## Layout

```
src/gsq_mlx/   — GSQ: grids, Gumbel-Softmax, local-shift, Lion, block loop
src/rco_mlx/   — RCO: manifold (project/retract/transport), DP knapsack, main loop
scripts/smoke_test.py — numpy+mlx-compatible smoke test, no weights needed
```

## Setup (M3)

`mlx` needs Python ≤3.13 as of late 2026. System python is 3.14, so:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install mlx mlx-lm numpy tqdm transformers
python scripts/smoke_test.py
```

Code is written `try import mlx.core except -> numpy fallback` so the smoke
test runs even before `mlx` is installed.

## Strategy for tiny models (why this works on Mac)

* Full paper: 4096 seqs × 4096 tokens × 20 epochs × 8×H200. We shrink to:
  `128-256 seqs × 512-1024 tokens × 5-10 epochs`, single block-loop,
  `group_size=64/128`, `Lion lr=1e-4`.
* RCO search space for tiny models: `N=~50-150 layers/groups, K=3-4 options
  (e.g. 2/3/4/8-bit)` — DP is `O(N·K·B')`, trivial on M3 (~seconds-minutes).
* Train/allocate on MLX, evaluate ppl on WikiText-2 slice, export symmetric
  `scale + int codes` → `mlx-lm` / GGUF.

Tell me the model (e.g. SmolLM2-135M / Qwen2.5-0.5B / Llama-3.2-1B /
TinyStories) and I wire `scripts/quantize_<model>.py` next.

## Phase 1 — GSQ for MLX (Oct 2026, synthetic-only, no model weights loaded)

Memory rule (Mac 17GB, ~2.5GB free): nothing in Phase 1 touches
`models/Qwen2.5-0.5B/model.safetensors` (942MB). Only `config.json`
shapes + synthetic weights / tiny `mx` arrays are used; both test scripts
run in seconds.

New / ported files (GSQ side owns only these + `scripts/quantize_qwen.py`):

* `src/gsq_mlx/quant.py` — real MLX ops (`mx.random.gumbel`, `mx.softmax`,
  pure functions safe under `mx.grad`) with a numpy fallback so
  `scripts/smoke_test.py` still passes unmodified. Covers Alg.1
  (`sample_gumbel` / `gs_softmax` / `gumbel_softmax_sample`), Alg.2
  ternary mask+sign (`ternary_soft` / `ternary_hard`), the 2-bit grid
  `[-2,-1,0,1]`, and the local-shift decoder (`local_shift_values` /
  `local_shift_quantize_soft` / `local_shift_decode`, delta -2..2, Eq.6).
* `src/gsq_mlx/gptq.py` — per-group RTN/GPTQ-lite init, `group_size=64`,
  from `config.json` shapes only (Qwen2.5-0.5B linears) + synthetic
  weights; App-A logit init (alpha 6 / std 0.01, ternary Eq.3-4).
  Hessian-aware GPTQ init is Phase-2 work.
* `src/gsq_mlx/block_train.py` — skeleton with staged losses
  (`qk` → `vo` → `mlp`, paper within-block staging), Lion steps
  (lr 1e-4 logits / 2e-5 scales, wd 1.0, betas 0.9/0.95), grad
  accumulation over Gumbel resamples, and a synthetic-token calibration
  stub (real FineWeb-Edu/OpenThoughts wiring is Phase 2). Both a numpy
  path and a pure-MLX path (`linear_soft_loss_mx`, usable under
  `mx.value_and_grad`) are included.
* `src/gsq_mlx/lion.py` — Lion now handles numpy and `mx` arrays.
* `scripts/test_gsq_phase1.py` — proves soft-to-hard collapse as
  tau→0, shift indices in range, Lion decreasing the synthetic linear
  loss, ternary paths, and MLX parity + grad flow.
* `scripts/quantize_qwen.py` — Phase-1 stub: per-layer RTN stats from
  shapes + synthetic weights only; prints what Phase 2 will wire.

Hyperparameters (paper App G / Tables 7-9): tau linear 2→0.05, kappa
linear 100→500, group_size 64 (Phase 1; paper uses 128).

```bash
.venv/bin/python scripts/test_gsq_phase1.py  # ~1s, 15 checks
.venv/bin/python scripts/smoke_test.py        # unchanged, still green
.venv/bin/python scripts/quantize_qwen.py --bits 2 --group-size 64
```

Phase 2 (not started): real Qwen weights/activations, tokenizer
calibration tokens, frozen-prefix block loop, scale-only fine-tune,
export to `mlx-lm`/GGUF.

## Phase 1 — RCO for MLX (Oct 2026, synthetic-only, no model weights loaded)

Same memory rule: nothing touches `models/Qwen2.5-0.5B/model.safetensors`.
Only `config.json` dims + synthetic logits / tiny `mx` arrays; both RCO
test scripts run in seconds.

New / ported files (RCO side owns only these + `scripts/allocate_qwen.py`):

* `src/rco_mlx/manifold.py` — real MLX ops (`mx.softmax`, pure functions
  safe under `mx.grad`: `softmax` / `expected_cost` / `normal_vector` /
  `project_tangent` / `transport`) with a numpy fallback so
  `scripts/smoke_test.py` still passes unmodified. Covers Eq.1 (expected
  cost), Eq.2 (normal), Sec-2 tangent projection, Prop.3 binary-search
  retract (`|C-B|<1e-6` in float64; mx float32 hits ~4e-6/ulp), App-A.5
  momentum transport, Eq.4 DP knapsack (CPU-side per App B.1, hardened
  with shape/feasibility checks) plus the Sec-3.1/App-B.1 exact O(N log N)
  top-k fast path for 2-option keep/prune (uniform weights).
* `src/rco_mlx/rco.py` — Alg.1 loop skeleton: Gumbel perturb
  `h=(a+G)/tau`, DP-solve, STE via soft probs of the SAME perturbed
  logits (closed-form `J^T v`, matches `mx.value_and_grad`), project grad
  (kills normal STE bias, Prop.6), Adam-style step (lr 0.05–0.2, default
  0.1), retract, transport momentum. Exp tau schedule 1.0→0.01,
  g-sample averaging (default 4). `synthetic_kl` + `assemble_weights_stub`
  are synthetic-only; real `mlx_lm` KL + GPTQ-residual assembly is Phase 2.
* `scripts/test_rco_phase1.py` — proves retract exactness, projection
  kills the normal, DP == brute force (N=6 K=3), top-k == DP == brute
  force, 10-step loop holds `|C-B|<1e-6`, and MLX parity + grad flow.
* `scripts/allocate_qwen.py` — Phase-1 stub: 24 layers × 4 groups from
  `config.json` only, options [2,3,4,8], synthetic sensitivities, prints
  per-layer avg bits + DP decode (real KL wiring is Phase 2).

```bash
.venv/bin/python scripts/test_rco_phase1.py  # ~1s, 17 checks
.venv/bin/python scripts/smoke_test.py        # unchanged, still green
.venv/bin/python scripts/allocate_qwen.py --avg-bits 3.0
```
