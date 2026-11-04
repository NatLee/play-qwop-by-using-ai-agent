"""Settings for the game page, the environment and the screen regions we read.

Every number that describes the game (where the score is drawn, how big the
end-of-race panel is, which key restarts) lives here, so the rest of the code
never hard-codes pixel positions or URLs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rewards import RewardConfig

# Bennett Foddy's official HTML5 QWOP. The old ``/Athletics.html`` address
# redirects here. The game is always loaded from this page at runtime; nothing
# from it is copied into this repository.
GAME_URL = "https://www.foddy.net/legacy/Athletics.html"


@dataclass(frozen=True)
class Region:
    """A rectangle on the game canvas, as fractions of the canvas size.

    ``size`` is the (width, height) the browser scales the rectangle to before
    handing it over; ``None`` keeps the native resolution.
    """

    left: float
    top: float
    width: float
    height: float
    size: tuple[int, int] | None = None

    def to_pixels(self, canvas_width: int, canvas_height: int) -> list[int]:
        """Return ``[x, y, w, h, out_w, out_h]`` in canvas pixels."""
        x = round(self.left * canvas_width)
        y = round(self.top * canvas_height)
        w = max(1, round(self.width * canvas_width))
        h = max(1, round(self.height * canvas_height))
        out_w, out_h = self.size if self.size is not None else (w, h)
        return [x, y, w, h, out_w, out_h]


@dataclass(frozen=True)
class GameConfig:
    """How to open and drive the official game page."""

    url: str = GAME_URL
    headless: bool = True
    # The game advances only when we ask for frames (see clock.js). 60 frames
    # per game second is what a normal monitor gives a human player.
    fps: int = 60
    # Sleep between frames so a visible browser plays at normal speed.
    realtime: bool = False
    restart_key: str = "r"
    load_timeout_s: float = 60.0
    # Frames to run after a restart so the runner settles before we look.
    settle_frames: int = 2
    viewport: tuple[int, int] = (1100, 600)
    browser_args: tuple[str, ...] = ("--mute-audio",)

    # Screen regions, measured on the 640x400 game canvas.
    # The "0.3 metres" text centred at the top.
    score_region: Region = Region(0.25, 0.045, 0.5, 0.085)
    # The middle of the screen, where the start / game-over panels appear.
    panel_region: Region = Region(0.22, 0.27, 0.56, 0.48, size=(56, 48))
    # The dark result bar on the end-of-race panel: "12.3 metres" after a fall,
    # "100 metres in 75.4 seconds" lower down after a finish. Only its white
    # text counts as ink, so one region covers both layouts.
    result_region: Region = Region(0.16, 0.46, 0.68, 0.21)
    # What the agent sees: the runner and the track around it, without the
    # key icons and score at the top.
    observation_region: Region = Region(0.2, 0.15, 0.6, 0.84)
    # Full frame for GIFs and ``render()``.
    render_region: Region = Region(0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class EnvConfig:
    """Episode rules and observation format for :class:`~qwop_agent.env.QWOPEnv`."""

    game: GameConfig = field(default_factory=GameConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    # 6 frames at 60 fps: the agent picks new keys 10 times per game second.
    # Frames in between are cheap (they skip drawing, see clock.js), so a
    # bigger skip mostly buys more game time per second of training.
    frame_skip: int = 6
    observation_size: tuple[int, int] = (84, 84)
    finish_distance: float = 100.0
    # End the episode early when the runner stops making progress.
    stall_seconds: float = 12.0
    stall_progress: float = 0.5
    # The original 2020 agent needed 592 s for 100 m, so give slow but steady runners room.
    max_game_seconds: float = 900.0
    # A reading that jumps further than this in one step is a misread.
    max_distance_jump: float = 2.0
    render_size: tuple[int, int] | None = None

    @property
    def step_seconds(self) -> float:
        return self.frame_skip / self.game.fps
