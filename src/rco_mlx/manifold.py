"""RCO for MLX — budget manifold + DP knapsack (paper Sec 2-3).

C(a) = sum_i w_i <p_i, c>, p_i = softmax(a_i).
Ops: normal n_ik = w_i p_ik (c_k - E[c]); project; retract via binary search
on t: C(a + t*c_vec) monotone increasing; transport momentum.
"""
from __future__ import annotations
import numpy as np


def softmax(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def expected_cost(logits: np.ndarray, costs: np.ndarray, weights: np.ndarray) -> float:
    p = softmax(logits)
    return float((weights[:, None] * p * costs[None, :]).sum())


def normal_vector(logits: np.ndarray, costs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    p = softmax(logits)
    exp_c = (p * costs[None, :]).sum(axis=1, keepdims=True)
    return weights[:, None] * p * (costs[None, :] - exp_c)


def project_tangent(grad: np.ndarray, normal: np.ndarray) -> np.ndarray:
    denom = float((normal * normal).sum()) + 1e-12
    return grad - (float((grad * normal).sum()) / denom) * normal


def retract(logits: np.ndarray, costs: np.ndarray, weights: np.ndarray,
            budget: float, lo: float = -50.0, hi: float = 50.0, iters: int = 40) -> np.ndarray:
    """Binary search t* s.t. C(a + t*c) = B (paper Prop.3 monotonic)."""
    a = logits.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c = expected_cost(a + mid * costs[None, :], costs, weights)
        if c < budget:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    return a + t * costs[None, :]


def transport(m: np.ndarray, normal_new: np.ndarray) -> np.ndarray:
    denom = float((normal_new * normal_new).sum()) + 1e-12
    return m - (float((m * normal_new).sum()) / denom) * normal_new


def dp_knapsack(scores: np.ndarray, costs_int: np.ndarray, weights_int: np.ndarray,
                budget_int: int) -> np.ndarray:
    """Multiple-choice knapsack, paper Eq.4, O(N*K*B'). Returns one-hot z [N,K].

    scores[N,K], each row picks exactly one k, sum w_i*c_k <= B.
    Tiny-model sizes only; uses -inf DP.
    """
    N, K = scores.shape
    NEG = -1e100
    dp = np.full(budget_int + 1, NEG)
    dp[0] = 0.0
    choice = np.zeros((N, budget_int + 1), dtype=np.int32)
    prev = np.zeros((N, budget_int + 1), dtype=np.int32)
    for i in range(N):
        ndp = np.full(budget_int + 1, NEG)
        for b in range(budget_int + 1):
            best, bk, pb = NEG, 0, 0
            for k in range(K):
                c = int(weights_int[i] * costs_int[k])
                if b - c >= 0 and dp[b - c] > NEG / 2:
                    v = dp[b - c] + scores[i, k]
                    if v > best:
                        best, bk, pb = v, k, b - c
            ndp[b], choice[i, b], prev[i, b] = best, bk, pb
        dp = ndp
    b = int(np.argmax(dp))
    z = np.zeros((N, K))
    for i in range(N - 1, -1, -1):
        k = int(choice[i, b])
        z[i, k] = 1.0
        b = int(prev[i, b])
    return z
