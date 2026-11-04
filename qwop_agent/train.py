"""Train a PPO agent (Stable-Baselines3) on several browsers at once."""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import gymnasium as gym

from .config import EnvConfig
from .env import QWOPEnv

logger = logging.getLogger(__name__)

# Episode columns written next to SB3's own ones in ``*.monitor.csv``.
EPISODE_KEYS = ("distance", "game_seconds", "finished")
FRAME_STACK = 4


@dataclass(frozen=True)
class PPOSettings:
    """PPO hyper-parameters (the usual Atari-style values)."""

    n_steps: int = 256
    batch_size: int = 512
    n_epochs: int = 4
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class TrainSettings:
    run_dir: Path = Path("runs/qwop-ppo")
    n_envs: int = 8
    total_timesteps: int = 20_000_000
    max_hours: float | None = None
    checkpoint_every: int = 100_000
    stop_after_finishes: int = 3
    resume: Path | None = None
    seed: int = 0
    device: str = "auto"


def make_vec_env(n_envs: int, env_factory: Callable[[], gym.Env], monitor_file: Path | None):
    """``n_envs`` environments (one browser each, in subprocesses) with episode logging and frame stacking."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor

    env_fns = [env_factory] * n_envs
    vec = SubprocVecEnv(env_fns, start_method="spawn") if n_envs > 1 else DummyVecEnv(env_fns)
    vec = VecMonitor(vec, filename=str(monitor_file) if monitor_file else None, info_keywords=EPISODE_KEYS)
    return VecFrameStack(vec, n_stack=FRAME_STACK)


def race_callback(run_dir: Path, max_seconds: float | None, stop_after_finishes: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class RaceCallback(BaseCallback):
        """Log distances, keep the best model and stop when the job is done."""

        def __init__(self):
            super().__init__()
            self.best_distance = float("-inf")
            self.finishes = 0
            self.recent: deque[float] = deque(maxlen=100)
            self._started = time.monotonic()

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode" not in info:
                    continue
                distance = float(info["distance"])
                self.recent.append(distance)
                if info.get("finished"):
                    self.finishes += 1
                    logger.info("finished 100 m in %.1f s of game time", info["game_seconds"])
                if distance > self.best_distance:
                    self.best_distance = distance
                    self.model.save(run_dir / "best_model")
                    logger.info("new best distance %.1f m at step %d", distance, self.num_timesteps)
            if self.recent:
                self.logger.record("race/best_distance", self.best_distance)
                self.logger.record("race/mean_distance_100", sum(self.recent) / len(self.recent))
                self.logger.record("race/finishes", self.finishes)
            if self.finishes >= stop_after_finishes:
                logger.info("stopping: the agent finished %d races", self.finishes)
                return False
            if max_seconds is not None and time.monotonic() - self._started > max_seconds:
                logger.info("stopping: time budget used up")
                return False
            return True

    return RaceCallback()


def train(
    settings: TrainSettings,
    env_config: EnvConfig | None = None,
    ppo: PPOSettings | None = None,
    env_factory: Callable[[], gym.Env] | None = None,
) -> Path:
    """Train (or resume) and return the path of the final model.

    ``env_factory`` builds one environment; the default plays the real game
    with ``env_config``. Tests pass a fake one.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure

    env_config = env_config or EnvConfig()
    ppo = ppo or PPOSettings()
    run_dir = settings.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # Each start (or resume) of a run is a new segment with its own log files.
    segment = len(list(run_dir.glob("episodes-*.monitor.csv"))) + 1
    env_factory = env_factory or partial(QWOPEnv, env_config)
    vec = make_vec_env(settings.n_envs, env_factory, run_dir / f"episodes-{segment}")
    try:
        if settings.resume:
            model = PPO.load(settings.resume, env=vec, device=settings.device)
            logger.info("resumed from %s at step %d", settings.resume, model.num_timesteps)
        else:
            model = PPO("CnnPolicy", vec, seed=settings.seed, device=settings.device, verbose=0, **asdict(ppo))
        model.set_logger(configure(str(run_dir / "logs" / f"segment-{segment}"), ["stdout", "csv", "tensorboard"]))
        checkpoints = CheckpointCallback(
            save_freq=max(1, settings.checkpoint_every // settings.n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="qwop_ppo",
        )
        max_seconds = settings.max_hours * 3600 if settings.max_hours else None
        callbacks = CallbackList([checkpoints, race_callback(run_dir, max_seconds, settings.stop_after_finishes)])
        model.learn(settings.total_timesteps, callback=callbacks, reset_num_timesteps=not settings.resume)
        final = run_dir / "final_model.zip"
        model.save(final)
        return final
    finally:
        vec.close()
