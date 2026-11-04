from pathlib import Path

import pytest

import main as legacy
from qwop_agent.cli import build_parser


def test_new_cli_commands_and_defaults():
    parser = build_parser()
    train = parser.parse_args(["train"])
    assert (train.n_envs, train.checkpoint_every, train.stop_after_finishes) == (8, 100_000, 3)
    assert train.run_dir == Path("runs/qwop-ppo")
    evaluate = parser.parse_args(["evaluate", "model/qwop-ppo.zip", "--gif", "docs/a.gif", "--headed"])
    assert evaluate.model == Path("model/qwop-ppo.zip") and evaluate.headed and not evaluate.stochastic
    compare = parser.parse_args(["compare", "early.zip", "late.zip", "--gif", "c.gif"])
    assert compare.labels == ("early training", "trained")
    assert parser.parse_args(["random"]).episodes == 3
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_legacy_flags_keep_their_old_defaults():
    args = legacy.parse_args([])
    assert args.train is True and args.test_env is False
    assert args.model_path is None and args.model_save_path is None
    old = legacy.parse_args(
        ["--train", "False", "--model_path", "./model/x.zip", "--chrome_driver_path", "./webdriver/chromedriver"]
    )
    assert old.train is False and old.model_path == Path("./model/x.zip")


def test_legacy_test_env_plays_random_races(monkeypatch):
    calls = []
    monkeypatch.setattr("qwop_agent.evaluate.play_random", lambda episodes: calls.append(episodes))
    assert legacy.main(["--test_env", "True", "--chrome_driver_path", "ignored"]) == 0
    assert calls == [None]  # until interrupted, like the original loop
