"""Teach a reinforcement learning agent to play the official browser QWOP."""

from .config import GAME_URL, EnvConfig, GameConfig, Region
from .env import ACTIONS, QWOPEnv
from .rewards import RewardConfig

__all__ = ["ACTIONS", "GAME_URL", "EnvConfig", "GameConfig", "QWOPEnv", "Region", "RewardConfig"]
__version__ = "2.0.0"
