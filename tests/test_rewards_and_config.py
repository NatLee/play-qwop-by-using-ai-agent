import pytest

from qwop_agent.config import GAME_URL, EnvConfig, Region
from qwop_agent.rewards import RewardConfig, StepOutcome, shaped_reward

CONFIG = RewardConfig()


def test_progress_is_rewarded_metre_for_metre():
    reward = shaped_reward(StepOutcome(previous_distance=10.0, distance=10.3))
    assert reward == pytest.approx(0.3 - CONFIG.time_penalty)


def test_sliding_backwards_costs():
    assert shaped_reward(StepOutcome(5.0, 4.8)) == pytest.approx(-0.2 - CONFIG.time_penalty)


def test_standing_still_is_never_free():
    assert shaped_reward(StepOutcome(3.0, 3.0)) < 0


def test_fall_and_finish_are_one_off_terms():
    fall = shaped_reward(StepOutcome(3.0, 3.0, fell=True))
    finish = shaped_reward(StepOutcome(99.9, 100.1, finished=True))
    assert fall == pytest.approx(-CONFIG.time_penalty - CONFIG.fall_penalty)
    assert finish == pytest.approx(0.2 - CONFIG.time_penalty + CONFIG.finish_bonus)


def test_reward_weights_can_be_changed():
    config = RewardConfig(progress_weight=2.0, time_penalty=0.0, fall_penalty=0.0, finish_bonus=0.0)
    assert shaped_reward(StepOutcome(1.0, 1.5, fell=True), config) == pytest.approx(1.0)


def test_a_whole_race_adds_up_to_its_distance():
    # Over a race the progress terms telescope: the sum is the final distance gain.
    distances = [0.3, 0.5, 0.4, 1.2, 2.0, 1.9]
    total = sum(shaped_reward(StepOutcome(a, b)) for a, b in zip(distances, distances[1:], strict=False))
    steps = len(distances) - 1
    assert total == pytest.approx(distances[-1] - distances[0] - steps * CONFIG.time_penalty)


def test_region_to_pixels():
    # Score text region on the 640x400 canvas: x 160..480, y 18..52.
    assert Region(0.25, 0.045, 0.5, 0.085).to_pixels(640, 400) == [160, 18, 320, 34, 320, 34]
    assert Region(0.0, 0.0, 1.0, 1.0, size=(84, 84)).to_pixels(640, 400) == [0, 0, 640, 400, 84, 84]


def test_defaults_describe_the_official_game():
    config = EnvConfig()
    assert GAME_URL.startswith("https://www.foddy.net/")
    assert config.game.url == GAME_URL
    assert config.step_seconds == pytest.approx(config.frame_skip / config.game.fps)
    assert config.finish_distance == 100.0
