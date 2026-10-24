"""Helpers around mlx-lm's Qwen2 for the FP checkpoint in models/Qwen2.5-0.5B.

The model is loaded once and never duplicated. Code that changes linear weights
(quantizers, the RCO objective) installs them with swapped_weights(), which puts
the original arrays back on exit.
"""
from __future__ import annotations

from contextlib import contextmanager

import mlx.core as mx

from .data import MODEL_DIR

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")
ATTN_LINEARS = frozenset(LINEARS[:4])
N_LAYERS = 24


def load_fp_model():
    """mlx-lm Qwen2 Model with the checkpoint's bf16 weights.

    The LM head is tied to the token embedding; evaluate.py relies on that.
    """
    from mlx_lm.utils import load_model

    model, _ = load_model(MODEL_DIR)
    if not model.args.tie_word_embeddings:
        raise ValueError("evaluate.py assumes a tied LM head")
    return model


def linear_path(layer: int, name: str) -> str:
    """Module path of a linear, e.g. model.layers.3.self_attn.q_proj."""
    block = "self_attn" if name in ATTN_LINEARS else "mlp"
    return f"model.layers.{layer}.{block}.{name}"


def linear_groups(n_layers: int = N_LAYERS) -> list[tuple[int, str]]:
    """(layer, name) for every quantizable linear: 7 per block, 168 for 24."""
    return [(i, name) for i in range(n_layers) for name in LINEARS]


def _module(model, layer: int, name: str):
    obj = model
    for part in linear_path(layer, name).split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def get_linear_weight(model, layer: int, name: str) -> mx.array:
    """Weight matrix [out, in] of a linear (the bias is not included)."""
    return _module(model, layer, name).weight


def set_linear_weight(model, layer: int, name: str, W: mx.array) -> None:
    """Install W [out, in], cast to the checkpoint dtype so the forward matches
    the deployed bf16 model."""
    module = _module(model, layer, name)
    if W.shape != module.weight.shape:
        raise ValueError(f"{linear_path(layer, name)}: expected "
                         f"{module.weight.shape}, got {W.shape}")
    module.weight = W.astype(module.weight.dtype)


@contextmanager
def swapped_weights(model, weights: dict[tuple[int, str], mx.array]):
    """Temporarily install {(layer, name): W}. The original arrays are restored
    on exit, including when the body raises."""
    originals = {key: get_linear_weight(model, *key) for key in weights}
    try:
        for (layer, name), W in weights.items():
            set_linear_weight(model, layer, name, W)
        yield model
    finally:
        for (layer, name), W0 in originals.items():
            _module(model, layer, name).weight = W0


def final_hidden(model, tokens) -> mx.array:
    """Normed final hidden states [B, T, H] (model.model), before the LM head."""
    return model.model(mx.array(tokens))
