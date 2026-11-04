"""Gymnasium environment around the official QWOP page."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .page import GamePage, GamePageError, PlaywrightGamePage
from .rewards import StepOutcome, shaped_reward
from .vision import DistanceReader, RaceResult, panel_visible

logger = logging.getLogger(__name__)

# Q and W swing the thighs, O and P bend the calves. Pressing both keys of a
# pair cancels out, so each pair has three useful states: neither, one, other.
THIGH_KEYS = ("", "q", "w")
CALF_KEYS = ("", "o", "p")
ACTIONS: tuple[frozenset[str], ...] = tuple(frozenset(t + c) for t in THIGH_KEYS for c in CALF_KEYS)

_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def action_label(action: int) -> str:
    """``"QO"`` for action 4, ``"-"`` for "no keys"."""
    return "".join(sorted(ACTIONS[action])).upper() or "-"


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """``(h, w, 3)`` RGB to ``(h, w, 1)`` grayscale ``uint8``."""
    return (rgb.astype(np.float32) @ _GRAY_WEIGHTS).clip(0, 255).astype(np.uint8)[:, :, None]


class QWOPEnv(gym.Env):
    """Play QWOP in a browser, one key combination per step.

    * Action: one of 9 key combinations (:data:`ACTIONS`), held for
      ``frame_skip`` game frames.
    * Observation: a grayscale crop around the runner.
    * Reward: see :mod:`qwop_agent.rewards`.
    * The episode ends when the game shows its end screen (after a fall or at
      the finish), and is cut short when the runner stops making progress or
      runs out of time.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
        page: GamePage | None = None,
        reader: DistanceReader | None = None,
    ):
        self.config = config or EnvConfig()
        game = self.config.game
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": game.fps / self.config.frame_skip}
        self.page = page if page is not None else PlaywrightGamePage(game)
        self.reader = reader if reader is not None else DistanceReader()

        width, height = self.config.observation_size
        self.observation_space = spaces.Box(0, 255, shape=(height, width, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._regions = {
            "observation": replace(game.observation_region, size=self.config.observation_size),
            "score": game.score_region,
            "panel": game.panel_region,
        }
        if render_mode == "rgb_array":
            self._regions["frame"] = replace(game.render_region, size=self.config.render_size)
        self._result_regions = {"result": game.result_region}

        self._broken = False
        self._steps = 0
        self._distance = 0.0
        self._stall_mark = 0.0
        self._stall_step = 0
        self._misreads = 0
        self._result: RaceResult | None = None
        self._observation = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self._frame: np.ndarray | None = None

    # -- Gymnasium API -------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if self._broken:
            self.page.close()
            self._broken = False
        self.page.open()  # loads the game the first time only
        self.page.restart()
        shots = self.page.advance(self.config.game.settle_frames, self._regions)
        self._steps = 0
        self._misreads = 0
        self._result = None
        self._distance = self.reader.distance(shots["score"]) or 0.0
        self._stall_mark = self._distance
        self._stall_step = 0
        self._remember(shots)
        return self._observation, self._info(fell=False, finished=False)

    def step(self, action):
        config = self.config
        try:
            self.page.hold(ACTIONS[int(action)])
            shots = self.page.advance(config.frame_skip, self._regions)
        except GamePageError as error:
            # The browser died. End this episode; reset() reopens the page.
            logger.warning("game page failed, will reopen it: %s", error)
            self._broken = True
            info = self._info(fell=False, finished=False)
            info["error"] = str(error)
            return self._observation, 0.0, False, True, info

        self._steps += 1
        previous = self._distance
        self._read_distance(shots["score"])
        game_over = panel_visible(shots["panel"])
        if game_over:
            # The fall screen says "12.3 metres", the finish screen
            # "101 metres in 19.4 seconds". The finish screen is the game's own
            # verdict: the score at the top can still say 99.x m at that point.
            self._result = self._read_result()
        # If the panel cannot be read, being past the line still counts.
        finished = game_over and (
            (self._result is not None and self._result.seconds is not None) or self._distance >= config.finish_distance
        )
        fell = game_over and not finished
        terminated = game_over
        truncated = not terminated and (self._stalled() or self.game_seconds >= config.max_game_seconds)

        reward = shaped_reward(StepOutcome(previous, self._distance, fell=fell, finished=finished), config.reward)
        self._remember(shots)
        return self._observation, reward, terminated, truncated, self._info(fell=fell, finished=finished)

    def render(self) -> np.ndarray | None:
        return self._frame if self.render_mode == "rgb_array" else None

    def close(self) -> None:
        self.page.close()

    # -- helpers -------------------------------------------------------------

    @property
    def game_seconds(self) -> float:
        return self._steps * self.config.step_seconds

    def _read_distance(self, crop: np.ndarray) -> None:
        reading = self.reader.distance(crop)
        if reading is None or abs(reading - self._distance) > self.config.max_distance_jump:
            self._misreads += 1
            return
        self._distance = reading
        if reading >= self._stall_mark + self.config.stall_progress:
            self._stall_mark = reading
            self._stall_step = self._steps

    def _read_result(self) -> RaceResult | None:
        try:
            shots = self.page.advance(1, self._result_regions)
        except GamePageError:
            return None
        return self.reader.result(shots["result"])

    def _stalled(self) -> bool:
        idle_steps = self._steps - self._stall_step
        return idle_steps * self.config.step_seconds >= self.config.stall_seconds

    def _remember(self, shots: dict[str, np.ndarray]) -> None:
        self._observation = to_gray(shots["observation"])
        if "frame" in shots:
            self._frame = shots["frame"]

    def _info(self, *, fell: bool, finished: bool) -> dict[str, Any]:
        info: dict[str, Any] = {
            "distance": self._distance,
            "game_seconds": round(self.game_seconds, 3),
            "fell": fell,
            "finished": finished,
            "misreads": self._misreads,
        }
        if self._result is not None and self._result.seconds is not None:
            # The number on the finish screen. The HTML5 game rounds the race
            # time to whole seconds and divides it by ten before showing it.
            info["finish_screen_seconds"] = self._result.seconds
        return info
