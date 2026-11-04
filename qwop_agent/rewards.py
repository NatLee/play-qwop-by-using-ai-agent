"""Reward shaping for QWOP.

The game only tells us one number: how far the runner is from the start line.
The shaped reward turns that into a per-step signal:

* progress: metres gained this step (negative when sliding backwards). This
  is the main signal; over a whole race it adds up to the final distance.
* ``time_penalty``: a small cost per step, so standing still is never free and
  a faster race scores better than a slow one. The original project had the
  same idea with its "time bonus".
* ``fall_penalty``: paid once when the runner falls and the race ends. It is
  kept small on purpose: a large one teaches the agent to freeze in a safe
  pose instead of trying to move.
* ``finish_bonus``: paid once when the runner reaches 100 m.

Misread distances are filtered out by the environment before they get here
(see ``EnvConfig.max_distance_jump``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 1.0
    time_penalty: float = 0.002
    fall_penalty: float = 2.0
    finish_bonus: float = 50.0


@dataclass(frozen=True)
class StepOutcome:
    previous_distance: float
    distance: float
    fell: bool = False
    finished: bool = False


def shaped_reward(outcome: StepOutcome, config: RewardConfig | None = None) -> float:
    """Return the reward for one environment step."""
    config = config or RewardConfig()
    reward = config.progress_weight * (outcome.distance - outcome.previous_distance)
    reward -= config.time_penalty
    if outcome.fell:
        reward -= config.fall_penalty
    if outcome.finished:
        reward += config.finish_bonus
    return reward
