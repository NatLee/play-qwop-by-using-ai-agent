"""Watch an agent race: evaluate a model, record GIFs, compare two agents."""

from __future__ import annotations

import itertools
import logging
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .config import EnvConfig
from .env import QWOPEnv
from .gif import caption, side_by_side, write_gif
from .train import FRAME_STACK

logger = logging.getLogger(__name__)

GIF_RENDER_SIZE = (640, 400)

Policy = Callable[[np.ndarray], int]


@dataclass
class Race:
    distance: float
    game_seconds: float
    finished: bool
    fell: bool
    steps: int
    reward: float
    finish_screen_seconds: float | None = None
    frames: list[np.ndarray] = field(default_factory=list, repr=False)

    def better_than(self, other: Race) -> bool:
        """Finishing beats not finishing; then faster finishes and longer runs win."""
        if self.finished != other.finished:
            return self.finished
        if self.finished:
            return self.game_seconds < other.game_seconds
        return self.distance > other.distance

    def summary(self) -> str:
        if self.finished:
            screen = f" (finish screen: {self.finish_screen_seconds:g})" if self.finish_screen_seconds else ""
            return f"finished 100 m in {self.game_seconds:.1f} s{screen}"
        how = "fell" if self.fell else "stopped"
        return f"{how} at {self.distance:.1f} m after {self.game_seconds:.1f} s"


class FrameStack:
    """Stack the last ``n`` observations on the channel axis, like ``VecFrameStack``."""

    def __init__(self, n: int, first: np.ndarray):
        self._frames = deque([np.zeros_like(first)] * (n - 1) + [first], maxlen=n)

    def push(self, observation: np.ndarray) -> None:
        self._frames.append(observation)

    @property
    def observation(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1)


def run_race(env: QWOPEnv, policy: Policy, record: bool = False, max_steps: int | None = None) -> Race:
    observation, _ = env.reset()
    stack = FrameStack(FRAME_STACK, observation)
    frames = [env.render()] if record else []
    total, steps, info = 0.0, 0, {}
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(stack.observation))
        stack.push(observation)
        total += reward
        steps += 1
        if record:
            frames.append(env.render())
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    return Race(
        distance=info["distance"],
        game_seconds=info["game_seconds"],
        finished=info["finished"],
        fell=info["fell"],
        steps=steps,
        reward=total,
        finish_screen_seconds=info.get("finish_screen_seconds"),
        frames=frames,
    )


def load_policy(path: Path, device: str = "auto"):
    """Load a policy from a training checkpoint or from an exported policy file."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticCnnPolicy

    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a model saved by this project (expected a .zip)")
    with zipfile.ZipFile(path) as archive:
        is_checkpoint = "policy.pth" in archive.namelist()  # SB3's documented checkpoint layout
    if is_checkpoint:
        return PPO.load(path, device=device).policy
    return ActorCriticCnnPolicy.load(path, device=device)


def policy_actor(policy, deterministic: bool) -> Policy:
    def act(observation: np.ndarray) -> int:
        action, _ = policy.predict(observation, deterministic=deterministic)
        return int(action)

    return act


def _recording_config(env_config: EnvConfig) -> EnvConfig:
    return replace(env_config, render_size=env_config.render_size or GIF_RENDER_SIZE)


def evaluate(
    model: Path,
    episodes: int = 1,
    deterministic: bool = True,
    gif: Path | None = None,
    label: str | None = None,
    max_gif_bytes: int = 5_000_000,
    gif_speed: float = 1.0,
    export_policy: Path | None = None,
    env_config: EnvConfig | None = None,
) -> list[Race]:
    env_config = env_config or EnvConfig()
    policy = load_policy(model)
    if export_policy is not None:
        policy.save(export_policy)
        print(f"policy weights saved to {export_policy}")
    env = QWOPEnv(_recording_config(env_config), render_mode="rgb_array" if gif else None)
    races: list[Race] = []
    best: Race | None = None
    try:
        for episode in range(1, episodes + 1):
            race = run_race(env, policy_actor(policy, deterministic), record=gif is not None)
            print(f"race {episode}: {race.summary()} (reward {race.reward:.1f})")
            races.append(race)
            # Keep frames only for the best race so far.
            if best is None or race.better_than(best):
                if best is not None:
                    best.frames = []
                best = race
            else:
                race.frames = []
    finally:
        env.close()
    print(f"best: {best.summary()}")
    if gif is not None:
        frames = [caption(f, label) for f in best.frames] if label else best.frames
        width, stride = write_gif(frames, gif, env_config.step_seconds, max_gif_bytes, speed=gif_speed)
        print(f"GIF saved to {gif} ({gif.stat().st_size / 1e6:.1f} MB, {width}px, every {stride} step(s))")
    return races


def compare(
    left: Path,
    right: Path,
    labels: tuple[str, str],
    gif: Path,
    max_steps: int,
    max_gif_bytes: int = 5_000_000,
    env_config: EnvConfig | None = None,
) -> tuple[Race, Race]:
    """Race two models one after the other in the same browser and put them side by side."""
    env_config = env_config or EnvConfig()
    env = QWOPEnv(_recording_config(env_config), render_mode="rgb_array")
    races = []
    try:
        for model, label in zip((left, right), labels, strict=True):
            race = run_race(env, policy_actor(load_policy(model), True), record=True, max_steps=max_steps)
            print(f"{label}: {race.summary()}")
            race.frames = [caption(frame, label) for frame in race.frames]
            races.append(race)
    finally:
        env.close()
    frames = side_by_side(races[0].frames, races[1].frames)
    width, stride = write_gif(frames, gif, env_config.step_seconds, max_gif_bytes)
    print(f"GIF saved to {gif} ({gif.stat().st_size / 1e6:.1f} MB, {width}px, every {stride} step(s))")
    return races[0], races[1]


def play_random(episodes: int | None, seed: int = 0, env_config: EnvConfig | None = None) -> list[Race]:
    """Press random keys; a quick check that the browser and the game work.

    ``episodes=None`` keeps playing until interrupted.
    """
    env = QWOPEnv(env_config or EnvConfig())
    env.action_space.seed(seed)
    races = []
    numbers = itertools.count(1) if episodes is None else range(1, episodes + 1)
    try:
        for episode in numbers:
            race = run_race(env, lambda _: int(env.action_space.sample()))
            print(f"race {episode}: {race.summary()}")
            races.append(race)
    finally:
        env.close()
    return races
