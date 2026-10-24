"""Chunked next-token NLL and KL through the tied Qwen2.5 LM head.

Full-vocabulary logits are large (2048 tokens x 151936 x fp32 = 1.2 GB), so the
head is applied to `chunk` positions at a time. The matmul runs in the model dtype
(bf16), as in mlx-lm's model(tokens); logits are cast to fp32 before any softmax.

Memory: a chunk's temporaries (fp32 logits, softmax intermediates) are only freed
once the chunk is evaluated. nll_from_hidden, perplexity and
kl_value_and_grad_hidden therefore evaluate one chunk at a time. Measured on the
real model, above the 0.93 GiB of weights: NLL head 0.05 GiB per chunk at chunk=64
(0.22 GiB at 256); kl_value_and_grad_hidden 0.25 GiB at chunk=64. kl_from_hidden
is differentiable through MLX's lazy graph, which keeps every chunk alive until
backward: 1.16 GiB above the weights at 512 positions (chunk=64) and about 2.8 GiB
at 2048 positions (chunk=256). Use kl_value_and_grad_hidden for long inputs.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from .model import final_hidden

CHUNK = 64


def _logits(model, h: mx.array) -> mx.array:
    """Tied LM head: fp32 logits [..., vocab] = h @ E^T."""
    return model.model.embed_tokens.as_linear(h).astype(mx.float32)


def nll_from_hidden(model, h, targets, chunk: int = CHUNK):
    """Summed next-token NLL over positions, -log p(y_t | h_t).

    h: [B, T, H] normed hidden states. targets: [B, T] token ids, where
    targets[b, t] is the token that h[b, t] predicts. Returns (sum NLL as an
    fp32 scalar, B*T). Do not differentiate through it: each chunk is evaluated
    eagerly.
    """
    B, T, H = h.shape
    hf = h.reshape(B * T, H)
    tg = mx.array(targets).reshape(B * T, 1)
    total = mx.array(0.0)
    for a in range(0, B * T, chunk):
        L = _logits(model, hf[a:a + chunk])
        picked = mx.take_along_axis(L, tg[a:a + chunk], axis=-1)[:, 0]
        total = total + mx.sum(mx.logsumexp(L, axis=-1) - picked)
        mx.eval(total)  # free this chunk's logits before the next one
    return total, B * T


def perplexity(model, windows, chunk: int = CHUNK,
               progress: bool = False) -> dict:
    """Perplexity of token windows [N, S] under the GPTQ / mlx-lm protocol.

    Each window is fed as tokens[:-1] and scored against tokens[1:], so it
    contributes S-1 predictions and its first token is never scored. Windows
    run one at a time to bound memory. Returns dict(ppl, nll_per_token,
    n_tokens, n_windows, seq_len).
    """
    windows = np.asarray(windows)
    n_windows, seq_len = windows.shape
    total, n_tokens = 0.0, 0
    for i in range(n_windows):
        x = mx.array(windows[i:i + 1])
        h = final_hidden(model, x[:, :-1])
        s, n = nll_from_hidden(model, h, x[:, 1:], chunk)
        total += float(s)
        n_tokens += n
        mx.clear_cache()
        if progress and (i + 1) % 10 == 0:
            print(f"  ppl: {i + 1}/{n_windows} windows", flush=True)
    nll = total / n_tokens
    return {"ppl": math.exp(nll), "nll_per_token": nll, "n_tokens": n_tokens,
            "n_windows": n_windows, "seq_len": seq_len}


def _log_softmax(z: mx.array) -> mx.array:
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def _kl_sum(model, hq, hr):
    """Sum over rows of KL(p_ref || p_q), p = softmax(head(h))."""
    lq = _log_softmax(_logits(model, hq))
    lr = _log_softmax(_logits(model, hr))
    return mx.sum(mx.exp(lr) * (lr - lq))


def kl_from_hidden(model, h_q, h_ref, chunk: int = CHUNK) -> mx.array:
    """Mean over positions of KL(p_ref || p_q) between two hidden-state streams.

    h_q, h_ref: [B, T, H] with the same shape. The reference log-probs are
    recomputed per chunk from h_ref, so no full-vocab tensor is cached. Each chunk
    runs under mx.checkpoint, which stops backward from storing chunk logits, but
    the lazy graph still spans all chunks: peak memory grows with the number of
    positions. Use kl_value_and_grad_hidden for long inputs.
    """
    B, T, H = h_q.shape
    hq = h_q.reshape(B * T, H)
    hr = h_ref.reshape(B * T, H)
    chunk_kl = mx.checkpoint(lambda a, b: _kl_sum(model, a, b))
    total = mx.array(0.0)
    for a in range(0, B * T, chunk):
        total = total + chunk_kl(hq[a:a + chunk], hr[a:a + chunk])
    return total / (B * T)


def _kl_chunk_value_grad(model, hq, hr):
    """Sum over rows of KL(p_ref || p_q) and its gradient w.r.t. hq (unscaled).

    With z_q = E h_q, the gradient of the row KL with respect to the logits is
    q - p_ref, so dKL/dh_q = (q - p_ref) @ E, computed in the model dtype as the
    autodiff backward of the head would be.
    """
    zq = _logits(model, hq)
    zr = _logits(model, hr)
    lse_q = mx.logsumexp(zq, axis=-1, keepdims=True)
    lse_r = mx.logsumexp(zr, axis=-1, keepdims=True)
    q = mx.exp(zq - lse_q)
    p_r = mx.exp(zr - lse_r)
    value = mx.sum(p_r * (zr - lse_r - zq + lse_q))
    E = model.model.embed_tokens.weight
    grad = ((q - p_r).astype(E.dtype) @ E).astype(hq.dtype)
    return value, grad


def kl_value_and_grad_hidden(model, h_q, h_ref, chunk: int = CHUNK):
    """Mean KL(p_ref || p_q) and its gradient w.r.t. h_q, chunk by chunk.

    Each chunk's value and gradient are evaluated before the next chunk starts, so
    peak memory is one chunk's logits however many positions there are. Returns
    (kl: fp32 scalar, grad: array shaped like h_q). The gradient is that of the
    mean over the B*T positions; multiply by the upstream cotangent when chaining
    through a network with mx.vjp.
    """
    B, T, H = h_q.shape
    n = B * T
    hq = h_q.reshape(n, H)
    hr = h_ref.reshape(n, H)
    total = mx.array(0.0)
    grads = []
    for a in range(0, n, chunk):
        v, g = _kl_chunk_value_grad(model, hq[a:a + chunk], hr[a:a + chunk])
        total = total + v
        grads.append(g)
        mx.eval(total, g)
    grad = mx.concatenate(grads, axis=0).reshape(B, T, H) / n
    return total / n, grad


def evaluate_mlx_model(model_dir, eval_len: int = 2048, max_windows: int | None = None,
                       progress: bool = True) -> dict:
    """WikiText-2 test perplexity of an exported mlx-lm checkpoint (mlx_lm.load), with its file size
    and the MLX peak memory: the GPTQ / mlx-lm protocol of perplexity() on eval_windows."""
    from mlx_lm import load
    from gsq_rco.common.data import eval_windows
    t0 = time.time()
    model, _ = load(str(model_dir))
    r = perplexity(model, eval_windows(seq_len=eval_len, split="test", max_windows=max_windows),
                   progress=progress)
    r.update(model=str(model_dir), seconds=time.time() - t0,
             file_bytes=sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors")),
             peak_mlx_gb=mx.get_peak_memory() / 2**30)
    return r
