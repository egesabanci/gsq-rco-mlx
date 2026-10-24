"""Checks for the RCO allocation on the quantized database (WP6, paper App B).

Usage: .venv/bin/python tests/test_rco_alloc.py   (exit 1 if any check fails)

A randomly initialised 2-block Qwen2 (the mlx-lm class the real model uses, hidden 64) and
random token windows, so the file runs in seconds without models/ or any download. The real
model is exercised by gsq-rco allocate (run under heavy.sh, see the WP6 report).
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm.models.qwen2 import Model, ModelArgs

from gsq_rco.gsq.gptq import dequantize, quantize_rtn
from gsq_rco.common import evaluate
from gsq_rco.common.model import LINEARS, get_linear_weight
from gsq_rco.rco import allocate as A
from gsq_rco.rco import database as D

RESULTS: list[tuple[str, bool]] = []
N_LAYERS, OPTIONS, GROUP = 2, (2, 3, 4), 32
VOCAB = 97


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}", flush=True)


def tiny_model(dtype=mx.float32, seed: int = 0):
    mx.random.seed(seed)
    args = ModelArgs(model_type="qwen2", hidden_size=64, num_hidden_layers=N_LAYERS,
                     intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
                     rms_norm_eps=1e-6, vocab_size=VOCAB, rope_theta=10000.0,
                     tie_word_embeddings=True)
    model = Model(args)
    if dtype != mx.float32:
        model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    return model


def fingerprint(model) -> str:
    h = hashlib.sha256()
    for _, v in sorted(tree_flatten(model.parameters()), key=lambda kv: kv[0]):
        h.update(np.asarray(v.astype(mx.float32)).tobytes())
    return h.hexdigest()


def linear_snapshot(model) -> list:
    return [np.asarray(get_linear_weight(model, L, n).astype(mx.float32))
            for L in range(N_LAYERS) for n in LINEARS]


class DenseBank:
    """Every option is the FP weight itself, so Delta_ik = 0 (the KL = 0 check)."""

    def __init__(self, model, options=OPTIONS):
        self.options = tuple(options)
        self.n_layers = N_LAYERS
        self.groups = [(L, n) for L in range(N_LAYERS) for n in LINEARS]
        self.counts = np.array([get_linear_weight(model, L, n).size for L, n in self.groups],
                               dtype=np.int64)
        self.sensitivity = np.zeros((len(self.groups), len(self.options)))
        self._W = [get_linear_weight(model, L, n).astype(mx.float32) for L, n in self.groups]

    def dequant(self, i, k):
        return self._W[i]


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(0)
    hess_tokens = rng.integers(0, VOCAB, size=(4, 24), dtype=np.int32)
    kl_tokens = rng.integers(0, VOCAB, size=(4, 16), dtype=np.int32)
    model = tiny_model()
    model_sha = fingerprint(model)
    before = linear_snapshot(model)

    # ---- dequantization matches the quantizer's own dequantize (ragged last group) ----
    W = rng.standard_normal((5, 100)).astype(np.float32)
    worst = 0.0
    for bits in (2, 3, 4, 8):
        r = quantize_rtn(W, bits, 64)
        ref = dequantize(r["codes"], r["scales"], bits, 64)
        got = np.asarray(D.dequant_mx(r["codes"], r["scales"], bits, 64))
        worst = max(worst, float(np.max(np.abs(ref - got))) / float(np.max(np.abs(ref))))
    check("dequant_mx == gptq.dequantize (bits 2,3,4,8)", worst < 1e-6, f"rel={worst:.1e}")

    # ---- database build: GPTQ on the tiny model, weights untouched ----
    db_dir = tempfile.mkdtemp(prefix="rco_db_test_")
    spec = D.DBSpec(options=OPTIONS, group_size=GROUP, quantizer="gptq", damp=0.01,
                    n_layers=N_LAYERS, calib_split="train", calib_seed=0)
    manifest = D.build_database(model, hess_tokens, spec, db_dir, model_sha,
                                log=lambda *_: None)
    unchanged = all(np.array_equal(a, b) for a, b in zip(before, linear_snapshot(model)))
    check("build leaves the FP weights unchanged", unchanged)
    db = D.Database(db_dir)
    n_groups = N_LAYERS * len(LINEARS)
    check("manifest shapes: 14 groups, 3 options, parameter counts",
          len(manifest["counts"]) == n_groups and db.sensitivity.shape == (n_groups, 3)
          and db.options == OPTIONS, f"counts[:7]={manifest['counts'][:7]}")
    sens_ok = bool(np.all(np.isfinite(db.sensitivity)) and np.all(db.sensitivity > 0))
    check("sensitivities are finite and positive", sens_ok,
          f"min={db.sensitivity.min():.3e} max={db.sensitivity.max():.3e}")
    codes_ok = True
    rt_worst = 0.0
    for i, (L, name) in enumerate(db.groups):
        for k, bits in enumerate(OPTIONS):
            codes = np.load(db.dir / D._name(L, name, bits, "codes"))
            scales = np.load(db.dir / D._name(L, name, bits, "scales"))
            codes_ok &= bool(codes.max() < 2 ** bits and codes.dtype == np.uint8)
            ref = dequantize(codes, scales, bits, GROUP)
            rt_worst = max(rt_worst, float(np.max(np.abs(np.asarray(db.dequant(i, k)) - ref))))
    check("codes in range and dequant(db) == gptq.dequantize of stored codes",
          codes_ok and rt_worst < 1e-6, f"max|diff|={rt_worst:.1e}")

    # ---- release=True (the lazy CLI build): same files, and FP weights freed as used ----
    released = tiny_model()
    rel_dir = tempfile.mkdtemp(prefix="rco_db_test_rel_")
    D.build_database(released, hess_tokens, spec, rel_dir, model_sha, log=lambda *_: None,
                     release=True)
    names = sorted(f for f in os.listdir(db.dir) if f.endswith(".npy"))
    same = all(np.array_equal(np.load(db.dir / f), np.load(os.path.join(rel_dir, f)))
               for f in names)
    freed = (released.model.embed_tokens.weight.size == 1
             and all(p.size == 1 for _, p in tree_flatten(released.model.layers[0].parameters())))
    check("release=True writes identical files and frees the FP weights as they are used",
          same and freed and len(names) == 2 * n_groups * len(OPTIONS) + 1,
          f"files={len(names)}")

    # ---- --db-quantizer rtn: codes are quantize_rtn of the FP weights, sensitivities finite ----
    rtn_dir = tempfile.mkdtemp(prefix="rco_db_test_rtn_")
    rspec = D.DBSpec(options=OPTIONS, group_size=GROUP, quantizer="rtn", n_layers=N_LAYERS)
    rtn = D.build_database(tiny_model(), hess_tokens, rspec, rtn_dir, model_sha,
                           log=lambda *_: None)
    W1 = np.asarray(get_linear_weight(tiny_model(), 1, "down_proj").astype(mx.float32))
    ref_codes = quantize_rtn(W1, OPTIONS[1], GROUP)["codes"]
    got_codes = np.load(os.path.join(rtn_dir, D._name(1, "down_proj", OPTIONS[1], "codes")))
    rtn_sens = np.load(os.path.join(rtn_dir, rtn["sensitivity_file"]))
    check("rtn database: codes == quantize_rtn(FP weights), sensitivities finite and positive",
          np.array_equal(got_codes, ref_codes) and rtn["damp"] is None
          and bool(np.all(np.isfinite(rtn_sens)) and np.all(rtn_sens > 0)))

    # ---- staleness: same inputs pass, changed model or tokens are refused ----
    try:
        db.verify(model_sha, hess_tokens)
        same_ok = True
    except D.StaleDatabase:
        same_ok = False
    check("verify passes for the same model and calibration tokens", same_ok)
    try:
        db.verify(model_sha, hess_tokens + 1)
        stale_tokens = False
    except D.StaleDatabase as e:
        stale_tokens = "calibration" in str(e)
    check("verify refuses changed calibration tokens", stale_tokens)
    try:
        db.verify("0" * 64, hess_tokens)
        stale_model = False
    except D.StaleDatabase as e:
        stale_model = "model_weights_sha256" in str(e)
    check("verify refuses changed model weights", stale_model)

    # ---- KL is zero when the database holds the FP weights ----
    dense = DenseBank(model)
    obj0 = A.KLObjective(model, dense, kl_tokens)
    choice0 = rng.integers(0, len(OPTIONS), size=n_groups)
    loss0, up0 = obj0.loss_and_upstream(choice0, [0, 1])
    check("KL(p_fp || p_z) == 0 and upstream == 0 when the database equals the FP weights",
          abs(loss0) < 1e-6 and float(np.max(np.abs(up0))) < 1e-6,
          f"loss={loss0:.2e} max|upstream|={np.max(np.abs(up0)):.2e}")

    # ---- assembled streamed forward equals the forward of the installed model ----
    obj = A.KLObjective(model, db, kl_tokens)
    choice = rng.integers(0, len(OPTIONS), size=n_groups)
    states = obj._states(kl_tokens[:1], lambda L: obj._installed(L, choice, None))
    streamed = np.asarray(obj.model.model.norm(states[-1]))
    installed = tiny_model()
    A.install_choice(installed, db, choice)
    direct = np.asarray(installed.model(mx.array(kl_tokens[:1])))
    check("streamed layer-by-layer assembly == forward of the installed model",
          float(np.max(np.abs(streamed - direct))) < 1e-5,
          f"max|diff|={np.max(np.abs(streamed - direct)):.2e}")
    check("model weights unchanged by the objective forward",
          all(np.array_equal(a, b) for a, b in zip(before, linear_snapshot(model))))

    # ---- STE upstream equals finite differences of L along Delta_ik ----
    loss, up = obj.loss_and_upstream(choice, [0])
    eps = 1e-2
    worst_rel = 0.0
    pairs = [(0, 1), (3, 2), (7, 0), (9, 1), (13, 2)]
    for i, k in pairs:
        L, name = db.groups[i]
        base = db.dequant(i, int(choice[i]))
        delta = db.dequant(i, k) - get_linear_weight(model, L, name).astype(mx.float32)
        lp = obj.mean_kl(choice, [0], override={i: base + eps * delta})
        lm = obj.mean_kl(choice, [0], override={i: base - eps * delta})
        fd = (lp - lm) / (2 * eps)
        rel = abs(fd - up[i, k]) / max(abs(fd), abs(up[i, k]), 1e-9)
        worst_rel = max(worst_rel, rel)
        print(f"    group {i:2d} option {k}: upstream={up[i, k]: .6e} fd={fd: .6e}")
    check("STE upstream == finite difference of L along Delta_ik (5 pairs)",
          worst_rel < 2e-2 and float(np.max(np.abs(up))) > 1e-6,
          f"worst rel err={worst_rel:.2e}")

    # ---- exact budget grid ----
    # tiny model: weights per layer [2,1,1,2,4,4,4] -> 18 units, 36 for two layers
    b3 = A.budget_grid(3.0, 36, 14)
    b2976 = A.budget_grid(2.976, 3120, 168)   # Qwen2.5-0.5B: 130 units per layer, 24 layers
    check("budget grid: integer average is ticks=1; 2.976 needs ticks=25 and is exact",
          b3 == (108.0, 1, 108) and b2976[1] == 25 and b2976[2] == 232128, f"{b3} {b2976}")
    try:
        A.budget_grid(0.1 + 0.2, 3120, 168)
        refused = False
    except ValueError:
        refused = True
    check("budget grid refuses an average whose decimal needs too many ticks", refused)

    # ---- short real loop: budget held every step, decode within the integer budget ----
    before_loop = linear_snapshot(model)
    res = A.rco_allocate(obj, db, 2.5, steps=3, g=2, lr=0.1, seed=0)
    hist = res["history"]
    gaps = [abs(h["cost"] - res["budget"]) for h in hist]
    check("RCO loop holds |C - B| < 1e-4 at every step", max(gaps) < 1e-4,
          f"gaps={[f'{g:.1e}' for g in gaps]}")
    check("decoded assignment is within the integer budget",
          res["decoded_int_cost"] <= res["budget_int"],
          f"decoded={res['decoded_int_cost']} budget_int={res['budget_int']}")
    check("loop records a finite loss per step",
          all(np.isfinite(h["loss"]) for h in hist), f"losses={[round(h['loss'], 4) for h in hist]}")
    check("model weights restored after the loop",
          all(np.array_equal(a, b) for a, b in zip(before_loop, linear_snapshot(model))))

    # ---- baselines and the install/evaluate path ----
    z_dp = A.proxy_dp_choice(db, 2.5)
    w_int = np.asarray(db.counts) // int(np.gcd.reduce(db.counts))
    dp_cost = int(np.sum(w_int * np.asarray(db.options)[z_dp]))
    check("proxy-DP baseline is feasible and uses the sensitivities",
          dp_cost <= int(round(2.5 * w_int.sum())), f"cost={dp_cost}")
    uni = A.uniform_choice(db, 3)
    check("uniform baseline decodes to exactly 3 bits on the grid",
          int(np.sum(w_int * np.asarray(db.options)[uni])) == 3 * int(w_int.sum()))
    try:
        A.uniform_choice(db, 5)
        refused = False
    except ValueError:
        refused = True
    check("uniform baseline refuses bits outside the options", refused)
    ppl = evaluate.perplexity(installed, kl_tokens)
    check("evaluate.perplexity runs on the installed allocation",
          np.isfinite(ppl["ppl"]) and ppl["n_windows"] == len(kl_tokens),
          f"ppl={ppl['ppl']:.3f}")
    per_linear, per_layer = A.allocation_summary(db, res["choice"])
    check("allocation summary covers every linear with a valid bit-width",
          len(per_linear) == n_groups and set(per_linear.values()) <= set(OPTIONS)
          and set(per_layer) == {str(L) for L in range(N_LAYERS)})

    failed = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed ({time.time() - t0:.1f}s)")
    if failed:
        print("FAILED: " + "; ".join(failed))
        return 1
    print("RCO ALLOCATION TESTS OK")
    return 0


def test_rco_alloc():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())


def test_quantized_from_database_reads_the_allocated_bits(tmp_path, monkeypatch):
    """`gsq-rco export --gptq-db`: each linear comes out at the bits its allocation gives it, with
    that option's codes and scales from the database. The model and calibration identity lookups
    (which read the real checkpoint) are patched to the tiny model's."""
    import json

    from gsq_rco.common.model import linear_path
    from gsq_rco.rco import driver
    model = tiny_model()
    model_sha = fingerprint(model)
    hess_tokens = np.random.default_rng(3).integers(0, VOCAB, size=(4, 24), dtype=np.int32)
    spec = D.DBSpec(options=OPTIONS, group_size=GROUP, quantizer="gptq", damp=0.01,
                    n_layers=N_LAYERS, calib_split="train", calib_seed=0)
    db_dir = str(tmp_path / "db")
    D.build_database(model, hess_tokens, spec, db_dir, model_sha, log=lambda *_: None)
    monkeypatch.setattr(driver, "model_fingerprint", lambda: model_sha)
    monkeypatch.setattr(driver, "calibration_tokens", lambda seqs, length, seed, split: hess_tokens)
    db = D.Database(db_dir)
    bits = {f"{L}.{name}": OPTIONS[(i * 2) % len(OPTIONS)] for i, (L, name) in enumerate(db.groups)}
    alloc = tmp_path / "alloc.json"
    alloc.write_text(json.dumps({
        "database": {"identity": {k: db.manifest[k] for k in
                                  ("format", "model_weights_sha256", "calibration", "hessian")}},
        "per_linear_bits": bits}))
    quantized = driver.quantized_from_database(db_dir, str(alloc), n_layers=N_LAYERS)
    assert len(quantized) == len(db.groups)
    for i, (L, name) in enumerate(db.groups):
        b = bits[f"{L}.{name}"]
        codes, scales = db.codes_scales(i, OPTIONS.index(b))
        q = quantized[linear_path(L, name)]
        assert q["bits"] == b and q["group_size"] == GROUP, (L, name)
        assert np.array_equal(q["codes"], codes) and np.array_equal(q["scales"], scales), (L, name)


def test_baseline_allocations_use_the_allocation_schema(tmp_path, monkeypatch):
    """`gsq-rco allocate --baseline uniform|proxy-dp` writes JSON that `eval --allocation` accepts:
    the database identity, per-linear bits on the database options, and the budget it was built for."""
    import json

    from gsq_rco.rco import driver
    model = tiny_model()
    model_sha = fingerprint(model)
    hess_tokens = np.random.default_rng(4).integers(0, VOCAB, size=(4, 24), dtype=np.int32)
    spec = D.DBSpec(options=OPTIONS, group_size=GROUP, quantizer="gptq", damp=0.01,
                    n_layers=N_LAYERS, calib_split="train", calib_seed=0)
    db_dir = str(tmp_path / "db")
    D.build_database(model, hess_tokens, spec, db_dir, model_sha, log=lambda *_: None)
    monkeypatch.setattr(driver, "model_fingerprint", lambda: model_sha)
    monkeypatch.setattr(driver, "calibration_tokens", lambda seqs, length, seed, split: hess_tokens)
    db = D.Database(db_dir)
    from types import SimpleNamespace
    for baseline, avg in (("uniform", 3), ("proxy-dp", 2.5)):
        out = tmp_path / f"{baseline}.json"
        driver.write_baseline(SimpleNamespace(db=db_dir, n_layers=N_LAYERS, baseline=baseline,
                                              avg_bits=avg), out)
        rec = json.loads(out.read_text())
        assert set(rec["per_linear_bits"]) == {f"{L}.{n}" for L, n in db.groups}
        assert all(b in OPTIONS for b in rec["per_linear_bits"].values())
        assert rec["database"]["identity"] == {k: db.manifest[k] for k in rec["database"]["identity"]}
        assert rec["checks"]["decode_ok"]
        if baseline == "proxy-dp":
            assert rec["decoded_avg_bits"] <= avg + 1e-9
        if baseline == "uniform":
            assert set(rec["per_linear_bits"].values()) == {3} and rec["decoded_avg_bits"] == 3.0
