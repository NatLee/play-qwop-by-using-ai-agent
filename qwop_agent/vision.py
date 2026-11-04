"""Read the game's state from canvas pixels.

QWOP keeps all of its state inside a closure, so the only things we can see
are the pixels it draws. Two readings matter:

* the distance text at the top ("12.3 metres"), read with Tesseract OCR;
* whether a panel covers the middle of the screen (the start screen, the
  "PARTICIPANT" fall screen and the finish screen all use one).

Tesseract stays loaded for the life of the process (tesserocr), so a reading
takes about 15 ms instead of the 100+ ms of starting the ``tesseract``
program each time. On top of that the game always draws the same string with
the same pixels, so :class:`DistanceReader` caches results by the exact text
mask and most steps are a dictionary lookup.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

_NUMBER = r"(-?\d+(?:\.\d+)?)"
DISTANCE_PATTERN = re.compile(_NUMBER + r"\s*m")
RESULT_PATTERN = re.compile(_NUMBER + r"\s*metres?(?:\s*in\s*" + _NUMBER + r"\s*s)?")

# Characters that can appear in "-12.3 metres in 45.6 seconds".
OCR_CHARACTERS = "-.0123456789metrsincod"

OcrFunction = Callable[[np.ndarray], str]


def text_mask(rgb: np.ndarray, threshold: int = 240) -> np.ndarray:
    """Return black text on a white background for light text in ``rgb``.

    The game writes its numbers in white (or near-white) on darker colours, so
    the darkest channel of a text pixel is still bright.
    """
    darkest = rgb.min(axis=2)
    return np.where(darkest > threshold, 0, 255).astype(np.uint8)


def panel_visible(
    rgb: np.ndarray,
    min_fraction: float = 0.45,
    brightness: int = 200,
    max_chroma: int = 30,
) -> bool:
    """True when a light grey panel covers most of ``rgb``.

    During play the middle of the screen is sky, crowd, grass and the runner;
    the light grey panels only show up on the start, fall and finish screens.
    """
    pixels = rgb.astype(np.int16)
    darkest = pixels.min(axis=2)
    chroma = pixels.max(axis=2) - darkest
    light_grey = (darkest > brightness) & (chroma < max_chroma)
    return float(light_grey.mean()) >= min_fraction


RESULT_TEXT_THRESHOLD = 150


def longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """``(start, stop)`` of the longest unbroken stretch of True values."""
    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return None
    longest = int(np.argmax(stops - starts))
    return int(starts[longest]), int(stops[longest])


def dark_bar(rgb: np.ndarray, darkness: int = 90, min_width: float = 0.3) -> np.ndarray | None:
    """Cut out the dark result bar from a crop of the end-of-race panel.

    The bar's top and bottom edges are the only rows with one long unbroken
    dark stretch (at least ``min_width`` of the crop): its text never touches
    the edges, and lines of dark panel text are broken up by their letters.
    Everything between the first and last such row is the bar.
    """
    dark = rgb.max(axis=2) < darkness
    widest = np.array([(run[1] - run[0]) if (run := longest_run(row)) else 0 for row in dark])
    solid_rows = np.flatnonzero(widest >= min_width * dark.shape[1])
    if len(solid_rows) == 0:
        return None
    top, bottom = solid_rows.min(), solid_rows.max() + 1
    columns = longest_run(dark[solid_rows].mean(axis=0) >= 0.5)
    if columns is None:
        return None
    left, right = columns
    return rgb[top:bottom, left:right]


def parse_distance(text: str) -> float | None:
    """Parse ``"12.3 metres"`` (or OCR's ``"12.3metres"``) into metres."""
    match = DISTANCE_PATTERN.search(text.replace(",", "."))
    return float(match.group(1)) if match else None


@dataclass(frozen=True)
class RaceResult:
    """What the end-of-race panel says. Only the finish screen shows a time."""

    distance: float
    seconds: float | None = None


def parse_result(text: str) -> RaceResult | None:
    """Parse the result bar: ``"0 metres"`` or ``"100.2 metres in 75.4 seconds"``."""
    match = RESULT_PATTERN.search(text.replace(",", "."))
    if not match:
        return None
    seconds = float(match.group(2)) if match.group(2) else None
    return RaceResult(float(match.group(1)), seconds)


class Tesseract:
    """One-line OCR with an engine that stays loaded between calls.

    The English language data is found through ``tessdata`` or the
    ``TESSDATA_PREFIX`` environment variable (the Docker image sets it), and
    otherwise in tesserocr's default location.
    """

    def __init__(self, tessdata: str | None = None):
        self._tessdata = tessdata or os.environ.get("TESSDATA_PREFIX")
        self._api = None

    def __call__(self, mask: np.ndarray) -> str:
        from PIL import Image

        if self._api is None:
            import tesserocr

            options = {"psm": tesserocr.PSM.SINGLE_LINE}
            if self._tessdata:
                options["path"] = self._tessdata
            self._api = tesserocr.PyTessBaseAPI(**options)
            self._api.SetVariable("tessedit_char_whitelist", OCR_CHARACTERS)
        self._api.SetImage(Image.fromarray(mask))
        return self._api.GetUTF8Text()


class DistanceReader:
    """Turn crops of the game canvas into numbers, with an exact-match cache."""

    def __init__(self, ocr: OcrFunction | None = None, cache_size: int = 8192, threshold: int = 240):
        self._ocr = ocr if ocr is not None else Tesseract()
        self._threshold = threshold
        self._cache: OrderedDict[bytes, str] = OrderedDict()
        self._cache_size = cache_size
        self.hits = 0
        self.misses = 0

    def text(self, rgb: np.ndarray, threshold: int | None = None) -> str:
        mask = text_mask(rgb, self._threshold if threshold is None else threshold)
        key = hashlib.blake2b(mask.tobytes(), digest_size=16).digest() + bytes(str(mask.shape), "ascii")
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            self._cache.move_to_end(key)
            return cached
        self.misses += 1
        text = self._ocr(mask).strip() if (mask == 0).any() else ""
        self._cache[key] = text
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return text

    def distance(self, rgb: np.ndarray) -> float | None:
        """Distance in metres from the score crop, or ``None`` if unreadable."""
        return parse_distance(self.text(rgb))

    def result(self, rgb: np.ndarray) -> RaceResult | None:
        """Distance (and time, after a finish) from a crop around the result bar."""
        bar = dark_bar(rgb)
        if bar is None:
            return None
        # Inside the dark bar anything clearly lighter than the bar is text.
        return parse_result(self.text(bar, threshold=RESULT_TEXT_THRESHOLD))
