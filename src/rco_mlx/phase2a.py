"""Phase 2a: REAL-weight RCO sensitivity for Qwen2.5-0.5B on MLX.

Memory-critical design (box has ~70MB free, model.safetensors is 942MB):
  * float16/bf16 everywhere for storage + activations (weights stay BF16
    as stored; fp32 is used ONLY inside norm/softmax/silu accumulations,
    then cast back — standard practice, no fp32 tensors retained).
  * batch size 1: calibration sequences are processed one at a time.
  * calibration capped at 8 sequences x 256 tokens (flags enforce the cap).
  * layer-by-layer: FP inputs/outputs are spilled to a temp dir on disk
    (~458KB per layer-seq in fp16), never held as full activations.
  * weights are mx.load'ed once; the embedding is dropped after the FP
    collection pass and each layer's weights are popped after its
    sensitivity is scored, so resident memory only shrinks.
  * mx.clear_cache() between layers AND between bit options.
  * NO torch anywhere. Tokenizer via the `tokenizers` lib on the
    model's own tokenizer.json. No lm_head needed (tied embeddings, and
    we only score per-layer MSE — no logits are ever computed).

Method (per task spec):
  * For each of the 24 layers and each option in [2,3,4,8]: RTN-quantize
    the 7 linear weights groupwise (group_size=64, symmetric, ported
    small into rtn_quantize() — no dependency on teammate code) and score
    mean MSE vs the FP layer output on calibration activations.
  * cost_i = bits x per-layer param count (norms/embed stay BF16 and are
    excluded from the quantized count; noted in the JSON).
  * The 24x4 sensitivity table feeds rco_loop (linear expected-
    sensitivity surrogate, normalized per layer); DP decode is exact on
    the integer grid (weights in units of 1M params keep B' ~ 1073).
"""
from __future__ import annotations
import gc
import json
import os
import tempfile
import time

import numpy as np

import mlx.core as mx

from rco_mlx.rco import rco_loop

OPTIONS = [2, 3, 4, 8]
GROUP_SIZE = 64
CAL_SEQS_MAX = 8
CAL_TOKENS_MAX = 256
W_UNIT = 1e6  # DP weight unit: 1M params (keeps B' ~ 1k)

LIN_KEYS = ["q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"]


# --------------------------------------------------------------- config ---

def load_cfg(config_path: str) -> dict:
    with open(config_path) as f:
        c = json.load(f)
    return {
        "layers": c["num_hidden_layers"],
        "hidden": c["hidden_size"],
        "intermediate": c["intermediate_size"],
        "heads": c["num_attention_heads"],
        "kv_heads": c["num_key_value_heads"],
        "head_dim": c["hidden_size"] // c["num_attention_heads"],
        "kv_dim": (c["num_key_value_heads"]
                   * (c["hidden_size"] // c["num_attention_heads"])),
        "rope_theta": float(c.get("rope_theta", 10000.0)),
        "eps": float(c.get("rms_norm_eps", 1e-6)),
        "vocab": c["vocab_size"],
    }


# ---------------------------------------------------------- calibration ---

def _default_passages():
    return [
        ("The photoelectric effect puzzled physicists for years. When light of high enough frequency "
         "strikes a metal surface, electrons are ejected almost instantly, and the kinetic energy of "
         "each electron grows with the frequency of the light, not its brightness. Classical wave theory "
         "predicted that brighter light should eventually eject electrons regardless of color, and that a "
         "dim beam would need time to deliver enough energy. Neither prediction held. Einstein proposed "
         "that light arrives in discrete packets, each carrying energy proportional to its frequency, so a "
         "single packet can liberate a single electron at once. The slope of stopping voltage against "
         "frequency gave a clean measurement of Planck's constant. Millikan verified the relation "
         "experimentally over a decade of painstaking work. The result united the particle and wave "
         "pictures of light and earned Einstein the Nobel Prize. It also seeded the idea that energy "
         "exchange at small scales is granular, a theme that would reshape chemistry, materials science, "
         "and our understanding of stellar spectra within a generation."),
        ("The fall of Constantinople in 1453 closed a chapter of history that had lasted more than a "
         "millennium. The Byzantine Empire, heir to Rome, had survived sieges, plagues, and schisms, but "
         "by the fifteenth century it held little beyond the city itself. Sultan Mehmed II assembled a "
         "vast force and, crucially, enormous bombards cast by the founder Orban, whose stones battered "
         "the Theodosian walls for weeks. The defenders, led by Constantine XI, numbered only a few "
         "thousand, including Genoese volunteers under Giustiniani. A small postern gate left open during "
         "the final assault allowed Ottoman troops inside, and resistance collapsed within hours. The "
         "conquest made Constantinople the Ottoman capital, redirected trade routes, and sent Greek "
         "scholars westward carrying manuscripts that would fuel the Renaissance. Historians still debate "
         "whether the city's fall or Columbus's voyage four decades later marks the sharper break "
         "between the medieval and modern worlds."),
        ("To make a proper risotto, start with hot stock kept at a bare simmer in a separate pot, since "
         "cold liquid shocks the rice and ruins the creamy texture. Soften finely diced onion in butter "
         "and olive oil until translucent, then add arborio rice and toast it until the grains turn "
         "pearly at the edges. Deglaze with a glass of dry white wine and let it absorb completely. Add "
         "stock one ladle at a time, stirring often but not constantly, waiting for each addition to be "
         "nearly absorbed before the next. The rice is done when al dente with a flowing, wave-like "
         "consistency called all'onda. Off the heat, beat in cold butter and grated parmesan for the "
         "mantecatura, which gives the dish its signature gloss. Rest two minutes, then serve on warm "
         "plates. Common mistakes include rinsing the rice, which washes away the surface starch that "
         "thickens the sauce, and cooking over fierce heat, which leaves the grains hard in the center "
         "while mushy outside."),
        ("A binary search tree keeps keys in sorted order by enforcing a simple invariant: every node in "
         "the left subtree holds a smaller key, and every node in the right subtree holds a larger one. "
         "Lookup walks down from the root, choosing left or right at each step, touching only as many "
         "nodes as the tree is deep. Insertion follows the same path and attaches the new node where the "
         "search falls off. Deletion is trickier: removing a leaf is trivial, a node with one child is "
         "spliced out, and a node with two children is replaced by its in-order successor. Without "
         "balancing, adversarial insertion orders degenerate the tree into a linked list, so self-balancing "
         "variants like red-black trees recolor nodes and rotate subtrees after each update to guarantee "
         "logarithmic height. The amortized analysis shows that rotations cost only constant time on "
         "average, which is why balanced trees underpin databases, filesystems, and language runtimes "
         "where predictable latency matters more than raw throughput."),
        ("Consider the sum of the first n odd numbers. Writing out cases gives 1, 4, 9, 16, suggesting "
         "the total equals n squared. Proof by induction starts with the base case n equals 1, which "
         "holds trivially. Assume the claim for some k, so the sum through the kth odd number is k "
         "squared. The next odd number is 2k plus 1, and adding it yields k squared plus 2k plus 1, which "
         "factors as k plus 1 squared. The induction closes and the formula holds for all positive "
         "integers. Geometrically, each odd number forms an L-shaped layer of unit squares around the "
         "previous square, building n-by-n squares one shell at a time. This visual proof, attributed to "
         "the Pythagoreans, needs no algebra at all. The same layering idea generalizes to higher "
         "dimensions, where sums of certain figurate numbers produce cubes and higher powers."),
        ("The night train from Chiang Mai rattled south through darkness while rain streaked the windows. "
         "In the sleeper car, a vendor sold grilled sticky rice wrapped in banana leaves, and an elderly "
         "monk across the aisle counted prayer beads by the light of his phone. By dawn the limestone "
         "karsts of the central plains rose out of flooded paddies, egrets standing motionless in the "
         "shallows. At the station, tuk-tuk drivers shouted destinations while the smell of boat noodles "
         "drifted from a cart near the exit. The old city revealed itself slowly: temple bells, canal "
         "bridges, a market where durian was stacked like artillery shells beside pyramids of mangosteen. "
         "In the afternoon heat, the crowds thinned and the traveler found a quiet shophouse cafe, its "
         "walls covered with fading photographs of the neighborhood in decades past, and planned the next "
         "leg of the journey over iced coffee that arrived in a tall glass with a steel straw."),
        ("Mitochondria generate most of a cell's ATP through oxidative phosphorylation. Electrons from "
         "NADH and FADH2 travel down the respiratory chain, and the energy released pumps protons across "
         "the inner membrane, building an electrochemical gradient. ATP synthase lets protons flow back, "
         "using the current to fuse ADP and phosphate. Because mitochondrial DNA is inherited maternally "
         "and mutates faster than nuclear DNA, it serves as a molecular clock for tracing lineages. Defects "
         "in the chain cause disorders that strike energy-hungry tissues first: muscle, brain, and heart. "
         "Exercise stimulates mitochondrial biogenesis, literally building more power plants inside cells, "
         "which is part of why endurance training improves fatigue resistance. Recent work suggests the "
         "organelles also signal stress to the nucleus, coordinating repair programs that extend beyond "
         "energy production into immunity and aging itself."),
        ("Inflation targeting asks central banks to steer price growth toward a stated goal, often two "
         "percent a year. When inflation runs hot, raising policy rates makes borrowing dearer, cooling "
         "demand with a lag of a year or more. When growth stalls, cutting rates cheapens credit and "
         "supports spending. The difficulty is that rates act with long and variable lags, so policymakers "
         "must drive by looking through the windshield at forecasts rather than the rearview mirror of "
         "realized data. Supply shocks, like oil embargoes, push prices up while depressing output, "
         "forcing an ugly choice between the two halves of a dual mandate. Credibility matters: if firms "
         "and workers believe the target will hold, they set wages and prices accordingly, and the belief "
         "becomes self-fulfilling. Losing that anchor, as happened in the seventies, requires a painful "
         "recession to restore, a lesson modern frameworks encode as forward guidance."),
    ]


def build_cal_ids(tokenizer_path: str, n_seqs: int = CAL_SEQS_MAX,
                  tokens: int = CAL_TOKENS_MAX) -> np.ndarray:
    """Deterministic calibration ids [n_seqs, tokens] from fixed passages."""
    if n_seqs > CAL_SEQS_MAX or tokens > CAL_TOKENS_MAX:
        raise ValueError("calibration capped at 8x256 (memory rule)")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    out = []
    for text in _default_passages()[:n_seqs]:
        ids = tok.encode(text).ids
        while len(ids) < tokens:  # repeat to guarantee length, then cut
            ids = ids + tok.encode(text).ids
        out.append(ids[:tokens])
    return np.asarray(out, dtype=np.int32)


# ------------------------------------------------------------- RTN quant ---

def rtn_quantize(w, bits: int, group: int = GROUP_SIZE):
    """Symmetric groupwise round-to-nearest over the input dim.

    Small local port of the groupwise RTN idea (group_size=64): per-group
    scale = max|w| / (2^(b-1)-1), range [-(2^(b-1)), 2^(b-1)-1]. No
    dependency on teammate code. bf16 in/out; fp32 scale math.
    """
    out, inn = w.shape
    qmax = float(2 ** (bits - 1) - 1)
    wf = w.astype(mx.float32)
    pad = (-inn) % group
    if pad:
        wf = mx.concatenate([wf, mx.zeros((out, pad))], axis=1)
    g = wf.reshape(out, -1, group)
    amax = mx.max(mx.abs(g), axis=-1, keepdims=True)
    scale = amax / qmax
    q = mx.clip(mx.round(g / (scale + 1e-12)), -(qmax + 1), qmax)
    return (q * scale).reshape(out, -1)[:, :inn].astype(mx.bfloat16)


# ------------------------------------------------------- Qwen2 forward ---

_rsqrt = mx.rsqrt if hasattr(mx, "rsqrt") else (lambda t: 1.0 / mx.sqrt(t))


def rmsnorm(x, w, eps: float):
    xf = x.astype(mx.float32)
    v = mx.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * _rsqrt(v + eps) * w.astype(mx.float32)).astype(mx.bfloat16)


def rope_cache(T: int, head_dim: int, theta: float):
    freq = 1.0 / (theta ** (np.arange(0, head_dim, 2,
                                     dtype=np.float32) / head_dim))
    ang = np.outer(np.arange(T, dtype=np.float32), freq)  # [T, D/2]
    emb = np.concatenate([ang, ang], axis=-1)             # [T, D] NeoX style
    return (mx.array(np.cos(emb)).astype(mx.bfloat16),
            mx.array(np.sin(emb)).astype(mx.bfloat16))


def _rot_half(x):
    d = x.shape[-1] // 2
    return mx.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def causal_mask(T: int):
    m = np.triu(np.full((T, T), np.float32(-1e9)), k=1)
    return mx.array(m).astype(mx.bfloat16)


def layer_param_count(cfg: dict) -> int:
    h, kvd, m = cfg["hidden"], cfg["kv_dim"], cfg["intermediate"]
    return (h * h + h * kvd * 2 + h * h + h * m * 2 + m * h)


def split_layer(W: dict, i: int) -> dict:
    p = f"model.layers.{i}."
    attn = p + "self_attn."
    mlp = p + "mlp."
    return {
        "q_w": W[attn + "q_proj.weight"], "q_b": W[attn + "q_proj.bias"],
        "k_w": W[attn + "k_proj.weight"], "k_b": W[attn + "k_proj.bias"],
        "v_w": W[attn + "v_proj.weight"], "v_b": W[attn + "v_proj.bias"],
        "o_w": W[attn + "o_proj.weight"],
        "gate_w": W[mlp + "gate_proj.weight"],
        "up_w": W[mlp + "up_proj.weight"],
        "down_w": W[mlp + "down_proj.weight"],
        "ln1": W[p + "input_layernorm.weight"],
        "ln2": W[p + "post_attention_layernorm.weight"],
    }


def quant_layer(Lw: dict, bits: int) -> dict:
    Q = dict(Lw)  # norms + biases stay BF16; only the 7 linears quantize
    for k in ["q_w", "k_w", "v_w", "o_w", "gate_w", "up_w", "down_w"]:
        Q[k] = rtn_quantize(Lw[k], bits)
    return Q


def layer_forward(x, Lw: dict, cfg: dict, cos, sin, mask):
    """Minimal Qwen2 decoder layer, bf16 in/out. x: [T, H]."""
    H, n_h, n_kv, D = (cfg["hidden"], cfg["heads"],
                       cfg["kv_heads"], cfg["head_dim"])
    rep = n_h // n_kv
    T = x.shape[0]
    h = rmsnorm(x, Lw["ln1"], cfg["eps"])
    q = (h @ Lw["q_w"].T + Lw["q_b"]).reshape(T, n_h, D)
    k = (h @ Lw["k_w"].T + Lw["k_b"]).reshape(T, n_kv, D)
    v = (h @ Lw["v_w"].T + Lw["v_b"]).reshape(T, n_kv, D)
    qh = mx.transpose(q, (1, 0, 2))                      # [Hq, T, D]
    kh = mx.transpose(k, (1, 0, 2))                      # [Hkv, T, D]
    vh = mx.transpose(v, (1, 0, 2))
    c = mx.expand_dims(cos, 0)
    s = mx.expand_dims(sin, 0)
    qh = qh * c + _rot_half(qh) * s
    kh = kh * c + _rot_half(kh) * s
    kh = mx.broadcast_to(mx.expand_dims(kh, 2),
                         (n_kv, T, rep, D)).reshape(n_h, T, D)
    vh = mx.broadcast_to(mx.expand_dims(vh, 2),
                         (n_kv, T, rep, D)).reshape(n_h, T, D)
    sc = (qh @ mx.transpose(kh, (0, 2, 1))) * (D ** -0.5) + mask
    aw = mx.softmax(sc.astype(mx.float32), axis=-1).astype(mx.bfloat16)
    av = mx.transpose(aw @ vh, (1, 0, 2)).reshape(T, H)
    h = x + av @ Lw["o_w"].T
    h2 = rmsnorm(h, Lw["ln2"], cfg["eps"])
    g = h2 @ Lw["gate_w"].T
    u = h2 @ Lw["up_w"].T
    gf = g.astype(mx.float32)
    act = (gf * mx.sigmoid(gf) * u.astype(mx.float32)).astype(mx.bfloat16)
    return h + act @ Lw["down_w"].T


# ----------------------------------------------- FP collection + MSE ---

def _spill(arr_mx, path: str):
    """Eval a bf16 array and spill to disk as fp16 (never hold full acts)."""
    a16 = arr_mx.astype(mx.float16)
    mx.eval(a16)
    np.save(path, np.asarray(a16))
    del a16


def _load(path: str):
    return mx.array(np.load(path).astype(np.float32)).astype(mx.bfloat16)


def collect_fp(weights: dict, embed, cfg: dict, cal_ids: np.ndarray,
               layers, cache_dir: str, cos, sin, mask) -> dict:
    """Run the FP prefix per sequence (batch size 1), spilling each
    layer's input + output to cache_dir. Returns timing info."""
    t0 = time.time()
    per_seq = []
    for s in range(cal_ids.shape[0]):
        ts = time.time()
        ids = mx.array(cal_ids[s].astype(np.int32))
        # One live activation (256x896 bf16) walks the layers; copies spill.
        x = mx.take(embed, ids, axis=0)
        mx.eval(x)
        del ids
        for i in layers:
            Lw = split_layer(weights, i)
            _spill(x, os.path.join(cache_dir, f"in_{i}_{s}.npy"))
            x = layer_forward(x, Lw, cfg, cos, sin, mask)
            mx.eval(x)
            _spill(x, os.path.join(cache_dir, f"out_{i}_{s}.npy"))
            del Lw
        del x
        mx.clear_cache()
        gc.collect()
        per_seq.append(time.time() - ts)
    return {"seconds": time.time() - t0, "per_seq": per_seq}


def score_layer(weights: dict, cfg: dict, cache_dir: str, i: int,
                n_seqs: int, cos, sin, mask, options=OPTIONS) -> dict:
    """MSE vs FP for one layer x each bit option. Pops layer weights."""
    Lw = split_layer(weights, i)
    mses = {}
    for bits in options:
        t0 = time.time()
        Q = quant_layer(Lw, bits)
        mx.eval(*[Q[k] for k in
                  ["q_w", "k_w", "v_w", "o_w", "gate_w", "up_w", "down_w"]])
        vals = []
        for s in range(n_seqs):
            x = _load(os.path.join(cache_dir, f"in_{i}_{s}.npy"))
            ref = _load(os.path.join(cache_dir, f"out_{i}_{s}.npy"))
            yq = layer_forward(x, Q, cfg, cos, sin, mask)
            m = mx.mean(mx.square((yq - ref).astype(mx.float32)))
            mx.eval(m)
            vals.append(float(m.item()))
            del x, ref, yq, m
        mses[str(bits)] = {"mean": float(np.mean(vals)), "per_seq": vals,
                           "seconds": time.time() - t0}
        del Q
        mx.clear_cache()
        gc.collect()
    # drop this layer's weights: resident set only shrinks from here
    p = f"model.layers.{i}."
    for k in [k for k in list(weights.keys()) if k.startswith(p)]:
        del weights[k]
    mx.clear_cache()
    gc.collect()
    return mses


# ------------------------------------------------------------- allocate ---

def normalize_sens(S: np.ndarray) -> np.ndarray:
    mu = S.mean(axis=1, keepdims=True)
    return S / np.maximum(mu, 1e-30)


def make_sensitivity_upstream(Sn: np.ndarray):
    """Constant upstream dL/dp = normalized sensitivity (linear expected-
    sensitivity surrogate L = sum p*S). Same every sample: g-averaging is
    then vacuous but the Alg.1 structure (per-sample DP solve) is kept;
    per-sample quantized forwards (10x4x8 full-model) are deferred as too
    heavy for this box — noted in results JSON."""
    C = np.asarray(Sn, dtype=np.float64)

    def fn(z, p, step, sample, rng):
        return C
    return fn


def run_allocation(S: np.ndarray, counts: np.ndarray, avg_bits: float,
                   steps: int, g: int, lr: float, seed: int,
                   init: str = "greedy"):
    from rco_mlx.manifold import dp_knapsack
    from rco_mlx.rco import to_int_grid
    costs = np.array(OPTIONS, dtype=np.float64)
    Sraw = np.asarray(S, dtype=np.float64)
    w = np.asarray(counts, dtype=np.float64) / W_UNIT
    budget = float(avg_bits) * float(w.sum())
    Sn = normalize_sens(Sraw)
    if init == "greedy":
        # Heuristic init (paper-sanctioned: scores need not start on M):
        # exact min-sensitivity assignment on the DP integer grid, as a
        # sharp one-hot. The loop then refines/verifies on the manifold.
        ci0, wi0, Bi0 = to_int_grid(costs, w, budget, 1)
        zg = dp_knapsack(-Sraw, ci0, wi0, Bi0)
        logits = zg * 3.0 - 0.75
    else:  # init == "negS": soft init favoring low-sensitivity options
        logits = -Sn.copy()
    res = rco_loop(logits, costs, w, budget, steps=steps, g=g, lr=lr,
                   seed=seed, grad_fn=make_sensitivity_upstream(Sn))
    z = np.asarray(res["z_final"])
    per_layer_bits = (z * np.array(OPTIONS)[None, :]).sum(axis=1)
    final_cost = float(res["history"][-1]["cost"])
    return {
        "res": res, "per_layer_bits": per_layer_bits,
        "expected_avg_bits": float(
            np.asarray([h["cost"] for h in res["history"]])[-1] / w.sum()),
        "budget_gap": abs(final_cost - budget),
        "weights_m": w, "budget": budget,
    }
