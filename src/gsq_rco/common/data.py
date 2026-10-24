"""WikiText-2 (raw v1) text and Qwen2.5 token ids under the GPTQ protocol.

The text is the HF dataset Salesforce/wikitext (config wikitext-2-raw-v1, parquet),
pinned to DATASET_REVISION and cached under <repo>/data/, which is git-ignored.
A failed download raises; there is no synthetic fallback.

Protocol: a split is its rows joined by blank lines (GPTQ). Tokens come from the
Qwen2.5 tokenizer with no BOS and no other special tokens. A window of S tokens
gives S-1 scored next-token predictions; its first token is never scored.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = Path(__file__).resolve().parents[3]  # src/gsq_rco/common/data.py -> repo root
DATA_DIR = REPO / "data"
MODEL_DIR = REPO / "models" / "Qwen2.5-0.5B"
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SPLITS = ("train", "validation", "test")


def _split_parquet(split: str) -> Path:
    """Local parquet file of a split, downloaded on first use."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    name = f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
    if (DATA_DIR / name).exists():
        return DATA_DIR / name
    return Path(hf_hub_download(DATASET, name, repo_type="dataset",
                                revision=DATASET_REVISION, local_dir=DATA_DIR))


def split_rows(split: str) -> int:
    """Number of rows (documents) of a split, read from the parquet metadata."""
    return pq.read_metadata(_split_parquet(split)).num_rows


def load_wikitext2(split: str) -> str:
    """Split as one string: rows joined by blank lines (GPTQ protocol)."""
    rows = pq.read_table(_split_parquet(split), columns=["text"])
    return "\n\n".join(rows.column("text").to_pylist())


def get_tokenizer() -> Tokenizer:
    """Qwen2.5 tokenizer from models/Qwen2.5-0.5B/tokenizer.json."""
    return Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))


def tokenize_split(split: str) -> np.ndarray:
    """Token ids of a whole split as int32, cached as .npy beside the parquet.

    The cache name embeds a hash of tokenizer.json, so a changed tokenizer
    re-tokenizes instead of reusing stale ids.
    """
    parquet = _split_parquet(split)
    digest = hashlib.sha256(
        (MODEL_DIR / "tokenizer.json").read_bytes()).hexdigest()[:12]
    cache = parquet.with_name(f"{parquet.stem}.{digest}.tokens.npy")
    if not cache.exists():
        ids = get_tokenizer().encode(load_wikitext2(split),
                                     add_special_tokens=False).ids
        tmp = cache.with_name(cache.name + ".tmp.npy")
        np.save(tmp, np.asarray(ids, dtype=np.int32))
        os.replace(tmp, cache)
    return np.load(cache)


def _grid(tokens: np.ndarray, seq_len: int) -> np.ndarray:
    """Non-overlapping seq_len-token windows of a stream; the tail is dropped."""
    n = len(tokens) // seq_len
    return tokens[: n * seq_len].reshape(n, seq_len)


def _sample_rows(grid: np.ndarray, n_seqs: int, seed: int) -> np.ndarray:
    """Seeded draw of n_seqs distinct rows of grid (without replacement)."""
    if not 0 < n_seqs <= len(grid):
        raise ValueError(f"asked for {n_seqs} windows, only {len(grid)} exist")
    idx = np.random.default_rng(seed).choice(len(grid), size=n_seqs,
                                             replace=False)
    return grid[idx]


def calibration_windows(n_seqs: int, seq_len: int, seed: int = 0,
                        split: str = "train") -> np.ndarray:
    """n_seqs windows of seq_len tokens, drawn at random from the non-overlapping
    seq_len-token windows of the split. Returns int32 [n_seqs, seq_len]."""
    return _sample_rows(_grid(tokenize_split(split), seq_len), n_seqs, seed)


def eval_windows(seq_len: int = 2048, split: str = "test",
                 max_windows: int | None = None) -> np.ndarray:
    """Consecutive non-overlapping windows from the start of the split; the
    remainder is dropped (GPTQ / mlx-lm eval protocol, no BOS).
    Returns int32 [n, seq_len]."""
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be >= 1")
    grid = _grid(tokenize_split(split), seq_len)
    return grid if max_windows is None else grid[:max_windows]
