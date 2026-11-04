"""The official QWOP page in a headless Chromium, driven through Playwright.

One :class:`PlaywrightGamePage` loads the game once and keeps reusing that
page: restarting a race is a key press, not a reload. Keys go to the page
(``page.keyboard``), never to the operating system, so several browsers can
play side by side.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import GameConfig, Region
from .vision import panel_visible

CLOCK_SCRIPT = Path(__file__).with_name("clock.js").read_text(encoding="utf-8")

Frames = dict[str, np.ndarray]


class GamePageError(RuntimeError):
    """The browser or the game page stopped working."""


class GamePage(Protocol):
    """What the environment needs from a game page (a fake one in tests)."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    def hold(self, keys: frozenset[str]) -> None:
        """Hold exactly ``keys`` down from now on, releasing all others."""

    def advance(self, frames: int, regions: Mapping[str, Region]) -> Frames:
        """Run ``frames`` game frames, then return RGB crops of ``regions``."""

    def restart(self) -> None:
        """Release all keys and start a new race."""


@dataclass
class PlaywrightGamePage:
    config: GameConfig = field(default_factory=GameConfig)

    def __post_init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._page = None
        self._held: frozenset[str] = frozenset()
        self._canvas_size: tuple[int, int] = (0, 0)
        self._last_wall_time: float | None = None
        self.loads = 0

    @property
    def is_open(self) -> bool:
        return self._page is not None

    def open(self) -> None:
        if self.is_open:
            return
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright

        config = self.config
        timeout_ms = config.load_timeout_s * 1000
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=config.headless, args=list(config.browser_args))
            width, height = config.viewport
            page = self._browser.new_page(viewport={"width": width, "height": height})
            page.add_init_script(script=CLOCK_SCRIPT)
            page.goto(config.url, wait_until="load", timeout=timeout_ms)
            self.loads += 1
            canvas = page.wait_for_selector("canvas", timeout=timeout_ms)
            self._page = page
            self._canvas_size = tuple(
                page.evaluate("() => { const c = document.querySelector('canvas'); return [c.width, c.height]; }")
            )
            self._wait_for_start_screen()
            canvas.click()  # "click to begin"
            self.advance(config.settle_frames, {})
        except PlaywrightError as error:
            self.close()
            raise GamePageError(f"could not open the game at {config.url}: {error}") from error

    def _wait_for_start_screen(self) -> None:
        """Run frames until the game has loaded and shows its start panel."""
        deadline = time.monotonic() + self.config.load_timeout_s
        regions = {"panel": self.config.panel_region}
        while time.monotonic() < deadline:
            if panel_visible(self.advance(1, regions)["panel"]):
                return
            time.sleep(0.02)  # let downloads and decoding happen in real time
        raise GamePageError("the game did not show its start screen in time")

    def close(self) -> None:
        # Closing a browser that already died must not raise.
        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:  # noqa: BLE001
                pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:  # noqa: BLE001
                pass
        self._playwright = self._browser = self._page = None
        self._held = frozenset()

    def hold(self, keys: frozenset[str]) -> None:
        page = self._require_page()
        try:
            for key in sorted(self._held - keys):
                page.keyboard.up(key)
            for key in sorted(keys - self._held):
                page.keyboard.down(key)
        except Exception as error:  # noqa: BLE001
            raise GamePageError(f"sending keys failed: {error}") from error
        self._held = keys

    def advance(self, frames: int, regions: Mapping[str, Region]) -> Frames:
        page = self._require_page()
        width, height = self._canvas_size
        request = {name: region.to_pixels(width, height) for name, region in regions.items()}
        try:
            result = page.evaluate(
                "([frames, frameMs, regions]) => window.__qwopClock.advance(frames, frameMs, regions)",
                [frames, 1000.0 / self.config.fps, request],
            )
        except Exception as error:  # noqa: BLE001
            raise GamePageError(f"running game frames failed: {error}") from error
        if self.config.realtime:
            self._pace(frames)
        shots = result.get("shots", {})
        missing = set(request) - set(shots)
        if missing:
            raise GamePageError(f"the game canvas is gone (no capture for {sorted(missing)})")
        return {name: decode_rgba(shots[name], *request[name][4:]) for name in request}

    def restart(self) -> None:
        self.hold(frozenset())
        try:
            self._require_page().keyboard.press(self.config.restart_key)
        except Exception as error:  # noqa: BLE001
            raise GamePageError(f"restarting the race failed: {error}") from error

    def _pace(self, frames: int) -> None:
        now = time.monotonic()
        if self._last_wall_time is not None:
            wait = frames / self.config.fps - (now - self._last_wall_time)
            if wait > 0:
                time.sleep(wait)
        self._last_wall_time = time.monotonic()

    def _require_page(self):
        if self._page is None:
            raise GamePageError("the game page is not open")
        return self._page


def decode_rgba(encoded: str, width: int, height: int) -> np.ndarray:
    """Turn the base64 RGBA bytes from clock.js into an ``(h, w, 3)`` RGB array."""
    pixels = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    return pixels.reshape(height, width, 4)[:, :, :3].copy()
