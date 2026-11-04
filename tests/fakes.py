"""Stand-ins for the browser and the OCR, so tests never load the real game."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from qwop_agent.config import Region
from qwop_agent.page import GamePageError
from qwop_agent.vision import RaceResult

PANEL_GREY = (237, 237, 237)
SKY = (40, 60, 90)
CANVAS = (640, 400)


def encode_distance(distance: float, shape: tuple[int, int, int]) -> np.ndarray:
    """A fake score crop that carries the distance in its first pixel."""
    crop = np.zeros(shape, dtype=np.uint8)
    decimetres = round(distance * 10) + 30000
    crop[0, 0, 0], crop[0, 0, 1] = divmod(decimetres, 256)
    crop[0, 0, 2] = 1  # "something is written here"
    return crop


@dataclass
class FakePage:
    """A runner whose distance follows ``timeline(frames since restart)``.

    The panel (fall or finish screen) shows from ``panel_from_frame`` on.
    """

    timeline: Callable[[int], float] = lambda frames: 0.3
    panel_from_frame: int | None = None
    fail_on_advance: int | None = None
    opened: bool = False
    loads: int = 0
    closes: int = 0
    restarts: int = 0
    frames: int = 0
    advances: int = 0
    held: frozenset[str] = frozenset()
    hold_log: list[frozenset[str]] = field(default_factory=list)

    def open(self) -> None:
        if not self.opened:
            self.opened = True
            self.loads += 1

    def close(self) -> None:
        self.opened = False
        self.closes += 1

    def hold(self, keys: frozenset[str]) -> None:
        self.held = keys
        self.hold_log.append(keys)

    def restart(self) -> None:
        self.held = frozenset()
        self.frames = 0
        self.restarts += 1

    def advance(self, frames: int, regions: Mapping[str, Region]) -> dict[str, np.ndarray]:
        if not self.opened:
            raise GamePageError("not open")
        self.advances += 1
        if self.fail_on_advance is not None and self.advances >= self.fail_on_advance:
            self.fail_on_advance = None
            raise GamePageError("browser crashed")
        self.frames += frames
        shots = {}
        for name, region in regions.items():
            x, y, w, h, out_w, out_h = region.to_pixels(*CANVAS)
            shape = (out_h, out_w, 3)
            if name == "score":
                shots[name] = encode_distance(self.timeline(self.frames), shape)
            elif name == "panel":
                shots[name] = np.full(shape, PANEL_GREY if self.panel_showing else SKY, dtype=np.uint8)
            else:
                value = (self.frames * 7) % 256
                shots[name] = np.full(shape, value, dtype=np.uint8)
        return shots

    @property
    def panel_showing(self) -> bool:
        return self.panel_from_frame is not None and self.frames >= self.panel_from_frame


class FakeReader:
    """Decodes :func:`encode_distance`; ``misread`` returns None for chosen readings."""

    def __init__(
        self,
        seconds: float | None = None,
        misread: Callable[[float], bool] = lambda d: False,
        readable_result: bool = True,
    ):
        self.seconds = seconds
        self.readable_result = readable_result
        self.misread = misread
        self.result_reads = 0

    def distance(self, crop: np.ndarray) -> float | None:
        if crop[0, 0, 2] == 0:
            return None
        distance = (int(crop[0, 0, 0]) * 256 + int(crop[0, 0, 1]) - 30000) / 10
        return None if self.misread(distance) else distance

    def result(self, crop: np.ndarray) -> RaceResult | None:
        """The panel reports a time only when this reader was given one."""
        self.result_reads += 1
        if not self.readable_result:
            return None
        return RaceResult(100.0 if self.seconds is not None else 1.0, self.seconds)
