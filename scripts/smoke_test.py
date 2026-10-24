"""Smoke test: GSQ soft/hard quant + RCO project/retract/DP. No model needed."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from gsq_mlx.quant import get_grid, gptq_init_logits, gumbel_softmax_sample, quantize_soft, quantize_hard
from gsq_mlx.lion import Lion
from rco_mlx.manifold import expected_cost, normal_vector, project_tangent, retract, dp_knapsack, softmax

rng = np.random.default_rng(0)
# --- GSQ: fake layer 256 weights, 2-bit ---
w = rng.standard_normal(256).astype(np.float32)
grid = get_grid(2)
logits = gptq_init_logits(w, grid)
s = float(np.abs(w).mean())
for t in range(5):
    tau = 2.0 - (2.0 - 0.05) * t / 4
    kappa = 100 + (500 - 100) * t / 4
    p = gumbel_softmax_sample(logits, tau, kappa, rng)
    wq = quantize_soft(p, grid, s)
    loss = float(((wq - w) ** 2).mean())
    print(f"gsq step {t} tau={tau:.2f} loss={loss:.4f}")
wh, idx = quantize_hard(logits, grid, s)
print("gsq hard mse:", float(((wh - w) ** 2).mean()))

# --- RCO: 20 groups x 4 bit-options, budget 2.6 avg ---
N, K = 20, 4
costs = np.array([2., 3., 4., 8.], dtype=np.float64)
weights = np.ones(N)
a = rng.standard_normal((N, K))
a = retract(a, costs, weights, budget=2.6 * N)
print("rco init cost:", expected_cost(a, costs, weights), "target:", 2.6 * N)
m = np.zeros_like(a)
for t in range(10):
    tau = max(0.01, 1.0 * (0.01) ** (t / 9))
    g = rng.standard_normal((N, K))  # surrogate for KL grad
    n = normal_vector(a, costs, weights)
    gt = project_tangent(g, n)
    a = a - 0.05 * gt  # Adam-lite
    a = retract(a, costs, weights, budget=2.6 * N)
    m = np.zeros_like(a)  # transport placeholder
print("rco final cost:", expected_cost(a, costs, weights))
z = dp_knapsack(a / max(tau, 1e-3), np.array([2, 3, 4, 8]), np.ones(N, dtype=int), int(2.6 * N))
print("rco decoded avg bit:", (z * costs[None, :]).sum() / N)
print("OK - ready for tiny model")
