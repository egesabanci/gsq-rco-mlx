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
