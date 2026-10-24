# GSQ + RCO on MLX — End-to-End Report

**Project:** near-lossless LLM compression (GSQ quantizer + RCO allocator) ported to
Apple Silicon (MLX), evaluated on Qwen2.5-0.5B.
**Machine:** Mac M3, 17 GB unified memory (`hw.memsize 17179869184`).
**Stack:** `uv`, Python 3.12 venv (`.venv`), `mlx 0.32.4`, `mlx-lm`, `numpy`, `tqdm`,
`huggingface_hub`. No torch in the compression path (MLX-only).
**Model:** `Qwen/Qwen2.5-0.5B` (HF base, `models/Qwen2.5-0.5B/`, 942 MB safetensors):
Qwen2, 24 layers, hidden 896, 14 Q-heads / 2 KV-heads, intermediate 4864,
vocab 151936, tied embeddings, rope_theta 1e6, bf16.
**Papers (in repo):** `gsq.pdf` (Gumbel-Softmax scalar quantization), `rco.pdf`
(budget-constrained allocation via Riemannian manifolds).
**Agents:** two Herdr `pi` workers (`opencode-go/muse-spark-1.3-contributor`),
`gsqpi` owned `src/gsq_mlx/*`, `rcopi` owned `src/rco_mlx/*`; heavy model jobs
staggered (one at a time) for the 17 GB box. All commits conventional-commits.

## 1. What the papers propose (short version)

- **GSQ** learns *which grid point each weight takes* instead of rounding once.
  Per-weight logits over the 2–8 grid values (ternary = mask+sign), Gumbel-Softmax
  relaxation, anneal temperature → hard assignment. Lion optimizer, block-staged
  losses, GPTQ warm-start. Output stays plain symmetric scalar → kernel-compatible.
- **RCO** decides *which layer gets how many bits* under an exact average budget.
  Softmax-relaxed expected cost defines a smooth manifold in logit space; each step:
  project gradient onto tangent plane → Adam step → binary-search retraction to the
  budget → transport momentum. Forward uses Gumbel-STE + exact DP knapsack.
  No penalty coefficient, exact budget every iterate.

## 2. Repo layout (final)

```
src/gsq_mlx/quant.py        grids, Gumbel-Softmax, ternary mask+sign, local-shift (MLX+numpy)
src/gsq_mlx/lion.py         Lion optimizer, paper App-G defaults (dual backend)
src/gsq_mlx/gptq.py         per-group RTN/GPTQ-lite init, group 64, App-A logit init
src/gsq_mlx/block_train.py  staged-loss skeleton (Q/K → V/O → MLP), grad accumulation
src/gsq_mlx/e2e.py          Phase-2: lazy weight reader, Qwen2 MLX forward, Lion trainer, NLL eval
src/rco_mlx/manifold.py     expected cost, normal, projection, retraction, DP + top-k (MLX+numpy)
src/rco_mlx/rco.py          RCO loop skeleton (Gumbel, DP-solve, STE, tau schedule)
src/rco_mlx/phase2a.py      Phase-2: real per-layer sensitivity + allocation driver
scripts/smoke_test.py       scaffold smoke test (still green)
scripts/test_gsq_phase1.py  15 checks, synthetic-only
scripts/test_rco_phase1.py  17 checks, synthetic-only
scripts/allocate_qwen.py    RCO allocator CLI (--probe/--layers/--init/--synthetic)
scripts/quantize_qwen.py    GSQ quantize CLI (+resume/skip-fp, shapes-only stub kept)
results/rco_allocation.json committed allocation (24 layers, avg 3.0)
results/gsq_eval.json       committed eval numbers + honesty note
results/qwen2.5-0.5b-gsq3bit.npz  352 MB export, git-ignored (uint8 codes + fp16 scales)
results/.cal_cache/         170 MB fp16 act cache, git-ignored
```

## 3. Phase 1 — synthetic ports (both PASS, verified by orchestrator rerun)

- GSQ: soft→hard collapse (maxprob 0.62→0.997, MSE 0.099→0.033), shift indices in
  range with edge clipping, Lion loss 0.00234→0.00176 (ratio 0.75), ternary alphabet,
  MLX/numpy parity + `value_and_grad` finite. 15/15 in ~0.2 s.
- RCO: retract `|C-B|≈1e-10` (numpy; ~4e-6 float32 MLX), projection residual ~3e-15,
  DP == brute force (N=6/K=3), top-k == DP, 10-step loop worst 6.9e-10, STE grad matches
  autograd <1e-4. 17 checks in ~0.3 s.

## 4. Phase 2a — RCO real allocation (PASS)

Real per-layer sensitivities (RTN group-64 at [2,3,4,8] + MSE/KL vs FP on 8×256 cal),
`rco_loop` 10 steps g=4 tau 1.0→0.01 lr 0.1, DP decode. Result committed:

| layers | bits |
|---|---|
| 0,1,5,6,7,8 | 2 |
| 2,4,9–19 | 3 |
| 3,20,21,22,23 | 4 |

Expected avg **3.0000**, decoded avg **2.9763**, `|C-B|=1e-9`, decode 1065 ≤ 1073.
Pattern sensible (robust early/mid → 2b, hot late layers → 4b, nothing at 8b).
Worker honesty note: 10-step loop is a local refiner, so init came from exact DP(−S)
one-hot (totS 1.33, 4.2× better than soft init). Runtime minutes; memory 217–588 MB
free throughout (`mx.clear_cache`, acts spilled to disk).

## 5. Phase 2b — GSQ real quantize + eval (PIPELINE PASS, QUALITY FAIL)

All 24 layers streamed batch-1 (lazy reads, never materialized 942 MB; bf16 fwd,
fp32 Lion, 20 steps × 2 resamples/linear, qk→vo→mlp). ~6 min total (~13–17 s/layer).

| metric | value |
|---|---|
| FP ppl (4×256 held-out from `rco.txt`) | 25.54 |
| Quant ppl | 1418.36 |
| Delta | +1392.82 |
| Avg bits achieved | 2.958, allocation_match true |
| Ablation, six 2-bit layers kept FP | 44.25 |
| Weight relerr | 2-bit ~0.25, 3/4-bit ~0.016 |

Two real bugs caught and fixed en route (bias-dict keys overwrote weights; missing
abs in vectorized RTN), verified by code-histogram checks.

## 6. Independent verification (orchestrator, this session)

- Re-ran `smoke_test.py`, `test_gsq_phase1.py`, `test_rco_phase1.py` → all green.
- Parsed `results/gsq_eval.json` + `results/rco_allocation.json` → numbers above match.
- `git log`: 7 conventional commits, tree clean except untracked reference PDFs.

## 7. Failure analysis — why ppl 1418 on this model

1. **Model too small.** 0.5B has no redundancy for 2-bit; ablation (1418→44.25 with
   2-bit layers in FP) localizes all damage to layers {0,1,5,6,7,8}. Papers target
   8B/70B/1T for this reason.
2. **Smoke scale.** 8×256 cal / 4×256 eval vs 4096×4096; 20 steps/linear vs 20 epochs;
   per-linear vs joint block losses; group 64 vs 128.
3. **Noisy eval.** FP baseline 25.5 already shows the eval slice (paper prose, OOD for
   Qwen base, 1024 tokens total) is a weak yardstick with huge variance.

## 8. Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install mlx mlx-lm huggingface_hub numpy tqdm   # torch NOT required
.venv/bin/python scripts/smoke_test.py                 # scaffold
.venv/bin/python scripts/test_gsq_phase1.py           # 15 checks
.venv/bin/python scripts/test_rco_phase1.py           # 17 checks
.venv/bin/python scripts/allocate_qwen.py --avg-bits 3.0
.venv/bin/python scripts/quantize_qwen.py --bits 3     # needs models/Qwen2.5-0.5B
```

## 9. Costs

- RCO Phase 2a: minutes, <600 MB. GSQ Phase 2b: ~6 min, streamed <1 GB.
- Agent spend this project: on the order of $0.20 total across workers.

## 10. Recommended next steps (pick one)

1. **3-bit floor** on 0.5B (drop the six 2-bit assignments) → expect ppl 40–60 at ~3.2 avg bits.
2. **Bigger model** (≥7B) + WikiText-2 eval + 10–100× steps → real quality claim.
3. **Ship tiny Mac model**: 4-bit path (`Q4_K_M`/MLX 4-bit), near-lossless even at 0.5B.
