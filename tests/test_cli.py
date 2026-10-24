"""gsq-rco command line: every subcommand parses, has examples in its help, and rejects bad combinations."""
import pytest

from gsq_rco import cli

COMMANDS = ["data", "eval", "build-db", "allocate", "quantize", "export", "run"]


@pytest.mark.parametrize("command", COMMANDS)
def test_help_has_examples(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "examples:" in out, command


def test_each_command_is_dispatched():
    parser = cli.build_parser()
    expected = {"data": "cmd_data", "eval": "cmd_eval", "build-db": "cmd_build_db",
                "allocate": "cmd_allocate", "quantize": "cmd_quantize", "export": "cmd_export",
                "run": "cmd_run"}
    for command, fn in expected.items():
        argv = {"data": ["data"], "eval": ["eval", "--fp"], "build-db": ["build-db"],
                "allocate": ["allocate", "--out", "x.json"], "quantize": ["quantize", "--uniform-bits", "3"],
                "export": ["export", "--out", "x"], "run": ["run", "--run-dir", "r"]}[command]
        assert parser.parse_args(argv).fn.__name__ == fn, command


def test_eval_takes_exactly_one_mode():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eval"])
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--fp", "--uniform", "3"])
    assert parser.parse_args(["eval", "--mlx-model", "m"]).mlx_model == "m"


def test_quantize_needs_alloc_or_uniform_bits():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["quantize", "--out", "x"])


def test_export_origins_are_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["export", "--out", "x", "--shards", "a", "--rtn-bits", "4"])


def test_run_shares_seed_and_names_rco_steps():
    ns = cli.build_parser().parse_args(["run", "--run-dir", "r", "--rco-steps", "7", "--steps", "9"])
    assert ns.rco_steps == 7 and ns.steps == 9 and ns.seed == 0
    assert ns.damp == 0.01 and ns.group_size == 64  # one value for build-db and quantize
