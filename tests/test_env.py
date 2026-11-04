from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from qwop_agent.config import EnvConfig
from qwop_agent.env import ACTIONS, QWOPEnv, action_label, to_gray
from qwop_agent.rewards import RewardConfig

from .fakes import FakePage, FakeReader

CONFIG = EnvConfig()
STEP = CONFIG.frame_skip  # frames per environment step
SETTLE = CONFIG.game.settle_frames  # frames run by reset()
TIME_PENALTY = CONFIG.reward.time_penalty


def after_steps(n: int) -> int:
    """Frames since the restart once ``n`` steps have run."""
    return SETTLE + n * STEP


def make_env(page=None, reader=None, config=CONFIG, **kwargs):
    page = page or FakePage()
    return QWOPEnv(config, page=page, reader=reader or FakeReader(), **kwargs), page


def walking(speed_per_frame: float, start: float = 0.3):
    return lambda frames: round(start + speed_per_frame * frames, 1)


def test_action_table_covers_each_thigh_and_calf_state_once():
    # 3 thigh states x 3 calf states; opposite keys of a pair never together.
    assert len(ACTIONS) == 9
    assert len(set(ACTIONS)) == 9
    assert frozenset() in ACTIONS
    for keys in ACTIONS:
        assert not {"q", "w"} <= keys
        assert not {"o", "p"} <= keys
    assert action_label(0) == "-"
    assert action_label(ACTIONS.index(frozenset("qo"))) == "OQ"


def test_reset_loads_the_game_once_and_returns_a_gray_crop():
    env, page = make_env()
    observation, info = env.reset(seed=1)
    assert observation.shape == (84, 84, 1)
    assert observation.dtype == np.uint8
    assert info["distance"] == pytest.approx(0.3)
    env.reset()
    env.reset()
    assert page.loads == 1  # the page is reused between races
    assert page.restarts == 3


def test_step_holds_the_chosen_keys():
    env, page = make_env()
    env.reset()
    for action, keys in enumerate(ACTIONS):
        env.step(action)
        assert page.held == keys
    assert page.frames == after_steps(len(ACTIONS))


def test_reward_is_progress_minus_time_cost():
    env, _ = make_env(FakePage(timeline=walking(0.1 / STEP)))  # 0.1 m per step
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == pytest.approx(0.1 - TIME_PENALTY)
    assert not terminated and not truncated
    assert info["game_seconds"] == pytest.approx(CONFIG.step_seconds)


def test_falling_ends_the_race_with_a_penalty():
    env, page = make_env(FakePage(panel_from_frame=after_steps(2)))
    env.reset()
    env.step(0)
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["fell"] and not info["finished"]
    assert reward == pytest.approx(-TIME_PENALTY - CONFIG.reward.fall_penalty)


def test_the_finish_screen_ends_the_race_with_bonus_and_its_time():
    reader = FakeReader(seconds=19.4)
    page = FakePage(timeline=lambda f: 99.9 if f < after_steps(2) else 100.2, panel_from_frame=after_steps(2))
    env, _ = make_env(page, reader)
    env.reset()
    env.step(0)
    _, reward, terminated, _, info = env.step(0)
    assert terminated
    assert info["finished"] and not info["fell"]
    assert info["finish_screen_seconds"] == 19.4
    assert reward == pytest.approx(0.3 - TIME_PENALTY + CONFIG.reward.finish_bonus)
    assert reader.result_reads == 1


def test_the_finish_screen_counts_even_if_the_score_shows_less_than_100_m():
    # The original 2020 agent's finish screen: the top said 99.4 m, the panel "100 metres in 592.1 seconds".
    reader = FakeReader(seconds=592.1)
    page = FakePage(timeline=lambda f: 99.4, panel_from_frame=after_steps(1))
    env, _ = make_env(page, reader)
    env.reset()
    _, reward, terminated, _, info = env.step(0)
    assert terminated and info["finished"] and not info["fell"]
    assert info["finish_screen_seconds"] == 592.1
    assert reward == pytest.approx(-TIME_PENALTY + CONFIG.reward.finish_bonus)


def test_a_fall_screen_without_a_time_is_a_fall():
    reader = FakeReader(seconds=None)
    env, _ = make_env(FakePage(panel_from_frame=after_steps(1)), reader)
    env.reset()
    _, _, terminated, _, info = env.step(0)
    assert terminated and info["fell"] and not info["finished"]
    assert "finish_screen_seconds" not in info
    assert reader.result_reads == 1


def test_crossing_100_m_waits_for_the_game_to_end_the_race():
    page = FakePage(timeline=lambda f: 100.2, panel_from_frame=after_steps(3))
    env, _ = make_env(page, FakeReader(seconds=19.4))
    env.reset()
    assert not env.step(0)[2]
    assert not env.step(0)[2]
    _, _, terminated, _, info = env.step(0)
    assert terminated and info["finished"]


def test_an_unreadable_end_screen_past_the_line_still_counts_as_a_finish():
    page = FakePage(timeline=lambda f: 100.1, panel_from_frame=after_steps(1))
    env, _ = make_env(page, FakeReader(readable_result=False))
    env.reset()
    _, _, terminated, _, info = env.step(0)
    assert terminated and info["finished"] and "finish_screen_seconds" not in info


def test_standing_still_is_cut_short():
    config = replace(CONFIG, stall_seconds=1.0)
    env, _ = make_env(config=config)
    env.reset()
    steps_to_stall = round(1.0 / config.step_seconds)
    for _ in range(steps_to_stall - 1):
        assert not env.step(0)[3]
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


def test_slow_but_steady_progress_is_not_a_stall():
    config = replace(CONFIG, stall_seconds=1.0)
    env, _ = make_env(FakePage(timeline=walking(1.0 / config.game.fps)), config=config)  # 1 m per game second
    env.reset()
    for _ in range(round(5.0 / config.step_seconds)):
        assert not env.step(0)[3]


def test_race_time_limit_truncates():
    config = replace(CONFIG, max_game_seconds=1.0, stall_seconds=99.0)
    env, _ = make_env(config=config)
    env.reset()
    results = [env.step(0) for _ in range(round(1.0 / config.step_seconds))]
    assert results[-1][3] and not any(r[3] for r in results[:-1])


def test_misreads_keep_the_last_good_distance():
    # One unreadable frame, then one wild jump (a misread "8" for "0.8").
    timeline = {STEP: 0.4, 2 * STEP: 0.5, 3 * STEP: 8.0, 4 * STEP: 0.7}
    reader = FakeReader(misread=lambda d: d == 0.5)
    env, _ = make_env(FakePage(timeline=lambda f: timeline.get(f - SETTLE, 0.3)), reader)
    env.reset()
    distances = [env.step(0)[4]["distance"] for _ in range(4)]
    assert distances == [0.4, 0.4, 0.4, 0.7]
    assert env.step(0)[4]["misreads"] == 2


def test_a_crashed_browser_ends_the_episode_and_reopens_on_reset():
    page = FakePage(fail_on_advance=3)
    env, _ = make_env(page)
    env.reset()
    env.step(0)
    _, reward, terminated, truncated, info = env.step(0)
    assert truncated and not terminated
    assert reward == 0.0
    assert "error" in info
    env.reset()
    assert page.closes == 1 and page.loads == 2


def test_render_returns_the_full_frame_only_in_rgb_array_mode():
    env, _ = make_env(render_mode="rgb_array", config=replace(CONFIG, render_size=(320, 200)))
    env.reset()
    env.step(0)
    assert env.render().shape == (200, 320, 3)
    plain, _ = make_env()
    plain.reset()
    assert plain.render() is None
    with pytest.raises(ValueError):
        QWOPEnv(CONFIG, render_mode="human", page=FakePage(), reader=FakeReader())


def test_custom_reward_weights_are_used():
    config = replace(CONFIG, reward=RewardConfig(progress_weight=2.0, time_penalty=0.0))
    env, _ = make_env(FakePage(timeline=walking(0.1 / STEP)), config=config)
    env.reset()
    assert env.step(0)[1] == pytest.approx(0.2)


def test_passes_the_gymnasium_checker():
    env, _ = make_env()
    check_env(env, skip_render_check=True)


def test_to_gray_uses_luminance_weights():
    # 0.299 R + 0.587 G + 0.114 B, truncated (the trained model saw exactly this).
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [100, 50, 200]]], dtype=np.uint8)
    gray = to_gray(rgb)
    assert gray.shape == (1, 4, 1)
    assert gray[0, :, 0].tolist() == [76, 149, 29, 82]  # 82.05 -> 82
