"""gsq-rco run: the step plan, chained config hashes, skip and run decisions, and database reuse.

The steps are replaced by fake runners, so no model or database is touched."""
import json
from pathlib import Path

import pytest

from gsq_rco import cli, run


def _args(tmp_path, *extra):
    argv = ["run", "--run-dir", str(tmp_path / "run"), "--db", str(tmp_path / "db")] + list(extra)
    return cli.build_parser().parse_args(argv)


def _fake_runners(calls):
    def make(name):
        def step(args, spec):
            calls.append(name)
            for p in spec["outputs"].values():
                Path(p).parent.mkdir(parents=True, exist_ok=True)
                Path(p).write_text("{}")
            return {"fake": name}
        return step
    return {name: make(name) for name in run.STEP_NAMES}


def test_plan_has_the_five_steps_in_order(tmp_path):
    names = [s["name"] for s in run.build_plan(_args(tmp_path))]
    assert names == list(run.STEP_NAMES)


def test_hash_chains_upstream_changes_downstream_only(tmp_path):
    base = {s["name"]: s["hash"] for s in run.build_plan(_args(tmp_path))}
    changed = {s["name"]: s["hash"] for s in run.build_plan(_args(tmp_path, "--avg-bits", "2.5"))}
    assert base["build-db"] == changed["build-db"]
    for name in ("allocate", "quantize", "export", "eval"):
        assert base[name] != changed[name], name
    later = {s["name"]: s["hash"] for s in run.build_plan(_args(tmp_path, "--max-windows", "4"))}
    assert later["allocate"] == base["allocate"] and later["eval"] != base["eval"]


def test_first_run_runs_all_then_second_run_skips_all(tmp_path):
    calls = []
    args = _args(tmp_path)
    run.run_pipeline(args, _fake_runners(calls))
    assert calls == list(run.STEP_NAMES)
    calls.clear()
    summary = run.run_pipeline(args, _fake_runners(calls))
    assert calls == []
    assert all(r == {"status": "skipped"} for r in summary["results"].values())


def test_changed_allocation_setting_reruns_from_allocate(tmp_path):
    calls = []
    run.run_pipeline(_args(tmp_path), _fake_runners(calls))
    calls.clear()
    run.run_pipeline(_args(tmp_path, "--avg-bits", "2.5"), _fake_runners(calls))
    assert calls == ["allocate", "quantize", "export", "eval"]


def test_dry_run_prints_plan_and_runs_nothing(tmp_path, capsys):
    calls = []
    out = run.run_pipeline(_args(tmp_path, "--dry-run"), _fake_runners(calls))
    assert calls == [] and out["dry_run"] is True
    assert [p["action"] for p in out["plan"]] == ["run"] * 5
    printed = json.loads(capsys.readouterr().out)
    assert [p["name"] for p in printed["plan"]] == list(run.STEP_NAMES)
    assert not (tmp_path / "run").exists()


def test_existing_database_is_reused_when_its_spec_matches(tmp_path, monkeypatch):
    from gsq_rco.rco import driver
    args = _args(tmp_path)
    step = next(s for s in run.build_plan(args) if s["name"] == "build-db")
    manifest = Path(step["outputs"]["manifest"])
    manifest.parent.mkdir(parents=True)
    c = step["config"]
    manifest.write_text(json.dumps({
        "options": c["options"], "group_size": c["group_size"], "quantizer": c["db_quantizer"],
        "damp": c["damp"], "n_layers": c["n_layers"],
        "calibration": {"n_windows": c["calib_seqs"], "window_len": c["calib_len"],
                        "seed": c["calib_seed"], "split": c["calib_split"]}}))
    monkeypatch.setattr(driver, "build_database_cmd",
                        lambda ns: pytest.fail("an existing database must not be rebuilt"))
    assert run._run_build_db(args, step) == {"status": "reused", "manifest": str(manifest)}


def test_database_with_other_settings_stops_the_run(tmp_path):
    args = _args(tmp_path)
    step = next(s for s in run.build_plan(args) if s["name"] == "build-db")
    manifest = Path(step["outputs"]["manifest"])
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"options": [2], "group_size": 64, "quantizer": "gptq",
                                    "damp": 0.01, "n_layers": 24,
                                    "calibration": {"n_windows": 16, "window_len": 2048,
                                                    "seed": 0, "split": "train"}}))
    with pytest.raises(SystemExit, match="other settings"):
        run._run_build_db(args, step)
