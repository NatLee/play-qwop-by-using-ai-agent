"""PPO training, checkpoints, evaluation, GIFs and the learning curve on a fake game."""

import csv

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("matplotlib")

from qwop_agent.config import EnvConfig  # noqa: E402
from qwop_agent.env import QWOPEnv  # noqa: E402
from qwop_agent.evaluate import FrameStack, load_policy, policy_actor, run_race  # noqa: E402
from qwop_agent.gif import write_gif  # noqa: E402
from qwop_agent.plot import load_episodes, plot_learning_curve, rolling_mean  # noqa: E402
from qwop_agent.train import FRAME_STACK, PPOSettings, TrainSettings, train  # noqa: E402

from .fakes import FakePage, FakeReader  # noqa: E402

TINY_PPO = PPOSettings(n_steps=32, batch_size=32, n_epochs=1)


def sprinting_env(render_mode=None):
    """A fake runner that covers 1.5 m per step whatever the keys, and finishes after ~67 steps."""
    config = EnvConfig(render_size=(64, 40))
    metres_per_frame = 1.5 / config.frame_skip
    page = FakePage(
        timeline=lambda frames: round(0.3 + frames * metres_per_frame, 1),
        panel_from_frame=round(100 / metres_per_frame),  # the finish screen
    )
    return QWOPEnv(config, render_mode=render_mode, page=page, reader=FakeReader(seconds=6.7))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run")
    settings = TrainSettings(
        run_dir=run_dir, n_envs=1, total_timesteps=2_000, checkpoint_every=64, stop_after_finishes=2, device="cpu"
    )
    final = train(settings, ppo=TINY_PPO, env_factory=sprinting_env)
    return run_dir, final


def test_training_stops_after_enough_finishes(run):
    run_dir, final = run
    episodes = load_episodes(run_dir)
    assert final.exists()
    assert sum(e.finished for e in episodes) == 2
    assert episodes[-1].step < 2_000


def test_training_writes_checkpoints_best_model_and_logs(run):
    run_dir, _ = run
    assert list((run_dir / "checkpoints").glob("qwop_ppo_*_steps.zip"))
    assert (run_dir / "best_model.zip").exists()
    assert list((run_dir / "logs").rglob("events.out.tfevents.*"))
    with next(run_dir.glob("episodes-1.monitor.csv")).open() as handle:
        handle.readline()
        header = next(csv.reader(handle))
    assert {"distance", "game_seconds", "finished"} <= set(header)


def test_resume_continues_in_a_new_segment(run, tmp_path):
    run_dir, final = run
    settings = TrainSettings(
        run_dir=run_dir, n_envs=1, total_timesteps=64, stop_after_finishes=99, resume=final, device="cpu"
    )
    train(settings, ppo=TINY_PPO, env_factory=sprinting_env)
    assert (run_dir / "episodes-2.monitor.csv").exists()
    steps = [e.step for e in load_episodes(run_dir)]
    assert steps == sorted(steps)


def test_checkpoints_and_exported_policies_both_load_and_race(run, tmp_path):
    _, final = run
    policy = load_policy(final, device="cpu")
    exported = tmp_path / "policy.zip"
    policy.save(exported)
    assert exported.stat().st_size < final.stat().st_size  # no optimiser state
    for model in (final, exported):
        env = sprinting_env(render_mode="rgb_array")
        race = run_race(env, policy_actor(load_policy(model, device="cpu"), deterministic=True), record=True)
        assert race.finished
        assert len(race.frames) == race.steps + 1
        assert "finished 100 m" in race.summary() and "finish screen: 6.7" in race.summary()


def test_load_policy_rejects_other_files(tmp_path):
    not_a_model = tmp_path / "model.h5"
    not_a_model.write_bytes(b"\x89HDF\r\n")
    with pytest.raises(ValueError, match="expected a .zip"):
        load_policy(not_a_model)


def test_frame_stack_matches_training_layout():
    first = np.full((84, 84, 1), 7, dtype=np.uint8)
    stack = FrameStack(FRAME_STACK, first)
    assert stack.observation.shape == (84, 84, FRAME_STACK)
    assert stack.observation[..., -1].max() == 7 and stack.observation[..., 0].max() == 0
    stack.push(np.full_like(first, 9))
    assert stack.observation[..., -1].max() == 9 and stack.observation[..., -2].max() == 7


def test_race_gif_and_learning_curve(run, tmp_path):
    run_dir, final = run
    env = sprinting_env(render_mode="rgb_array")
    race = run_race(env, policy_actor(load_policy(final, device="cpu"), deterministic=True), record=True)
    gif = tmp_path / "race.gif"
    write_gif(race.frames, gif, env.config.step_seconds, max_bytes=2_000_000)
    assert gif.stat().st_size > 0
    chart = plot_learning_curve(run_dir, tmp_path / "curve.png")
    assert chart.stat().st_size > 10_000


def test_rolling_mean():
    assert rolling_mean(np.array([1.0, 3.0, 5.0, 7.0]), window=2).tolist() == [1.0, 2.0, 4.0, 6.0]


def test_the_best_race_is_a_finish_then_the_fastest():
    from qwop_agent.evaluate import Race

    def race(distance, seconds, finished):
        return Race(distance, seconds, finished, fell=not finished, steps=1, reward=0.0)

    fast, slow, far, near = race(100.3, 190, True), race(100.9, 260, True), race(60, 300, False), race(20, 30, False)
    assert fast.better_than(slow) and not slow.better_than(fast)
    assert slow.better_than(far) and far.better_than(near)
