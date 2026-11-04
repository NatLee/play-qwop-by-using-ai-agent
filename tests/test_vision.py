import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from qwop_agent.vision import (
    DistanceReader,
    RaceResult,
    dark_bar,
    longest_run,
    panel_visible,
    parse_distance,
    parse_result,
    text_mask,
)

SKY = (52, 74, 99)
PANEL = (237, 237, 237)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.3metres", 0.3),  # what Tesseract returns for "0.3 metres"
        ("12.3 metres", 12.3),
        ("-1.2metres", -1.2),
        ("7metres", 7.0),
        ("1,5 metres", 1.5),
        ("metres", None),
        ("", None),
    ],
)
def test_parse_distance(text, expected):
    assert parse_distance(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("7.8 metres", RaceResult(7.8, None)),
        ("100.3 metres in 85.2 seconds", RaceResult(100.3, 85.2)),
        ("100.3metresin85.2seconds", RaceResult(100.3, 85.2)),
        ("everyone is a winner", None),
    ],
)
def test_parse_result(text, expected):
    assert parse_result(text) == expected


def test_text_mask_keeps_only_near_white_pixels():
    rgb = np.array([[[255, 255, 255], [250, 245, 241], [237, 237, 237], [255, 255, 0]]], dtype=np.uint8)
    # White and near-white text become black ink; the grey panel and yellow stay background.
    assert text_mask(rgb).tolist() == [[0, 0, 255, 255]]


def test_panel_detection():
    sky = np.full((48, 56, 3), SKY, dtype=np.uint8)
    assert not panel_visible(sky)
    # The runner's white shoes or bib are not a panel.
    sky_with_bib = sky.copy()
    sky_with_bib[10:20, 20:30] = 255
    assert not panel_visible(sky_with_bib)
    panel = np.full((48, 56, 3), PANEL, dtype=np.uint8)
    panel[20:28, 10:46] = (29, 43, 56)  # the dark result bar
    assert panel_visible(panel)


def strip_with_ink(columns):
    rgb = np.full((10, 40, 3), SKY, dtype=np.uint8)
    for column in columns:
        rgb[2:8, column] = 255
    return rgb


class CountingOcr:
    def __init__(self, text="1.5 metres"):
        self.text = text
        self.calls = 0

    def __call__(self, mask):
        assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {0, 255}
        self.calls += 1
        return self.text


def test_reader_caches_identical_crops():
    ocr = CountingOcr()
    reader = DistanceReader(ocr=ocr)
    assert reader.distance(strip_with_ink([3, 4])) == 1.5
    assert reader.distance(strip_with_ink([3, 4])) == 1.5
    assert ocr.calls == 1
    assert (reader.hits, reader.misses) == (1, 1)
    reader.distance(strip_with_ink([5, 6]))
    assert ocr.calls == 2


def test_reader_skips_ocr_when_nothing_is_written():
    ocr = CountingOcr()
    reader = DistanceReader(ocr=ocr)
    assert reader.distance(strip_with_ink([])) is None
    assert ocr.calls == 0


def test_reader_cache_is_bounded():
    ocr = CountingOcr()
    reader = DistanceReader(ocr=ocr, cache_size=2)
    for column in (1, 2, 3):
        reader.distance(strip_with_ink([column]))
    reader.distance(strip_with_ink([1]))  # evicted, so read again
    assert ocr.calls == 4


def render_score(text: str) -> np.ndarray:
    """White bold-ish text on the game's sky colour, like the score at the top."""
    image = Image.new("RGB", (320, 34), SKY)
    font = ImageFont.load_default(size=26)
    draw = ImageDraw.Draw(image)
    draw.text(
        (160, 17), text, fill=(255, 255, 255), font=font, anchor="mm", stroke_width=1, stroke_fill=(255, 255, 255)
    )
    return np.asarray(image)


def tesseract_available() -> bool:
    try:
        import tesserocr
    except ImportError:
        return False
    if os.environ.get("TESSDATA_PREFIX"):
        return True
    return bool(tesserocr.get_languages()[1])


@pytest.mark.skipif(not tesseract_available(), reason="needs tesserocr and English tessdata")
@pytest.mark.parametrize("text", ["0.3 metres", "12.7 metres", "-1.4 metres", "100.2 metres"])
def test_real_tesseract_reads_rendered_distances(text):
    reader = DistanceReader()
    assert reader.distance(render_score(text)) == pytest.approx(float(text.split()[0]))


BAR = (29, 43, 56)


def render_panel(text: str | None, bar_top: int = 20) -> np.ndarray:
    """A crop of the end-of-race panel: light grey, game background at the sides, a dark bar."""
    image = Image.new("RGB", (440, 84), (20, 30, 45))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 0, 410, 83), fill=PANEL)
    draw.text((220, 5), "everyone is a winner", fill=(40, 40, 40), font=ImageFont.load_default(size=14), anchor="mt")
    if text is not None:
        draw.rectangle((70, bar_top, 370, bar_top + 30), fill=BAR)
        font = ImageFont.load_default(size=18)
        draw.text((220, bar_top + 15), text, fill=(255, 255, 255), font=font, anchor="mm")
    return np.asarray(image)


def test_dark_bar_is_cut_out_of_the_panel():
    bar = dark_bar(render_panel("7.8 metres", bar_top=40))
    assert bar is not None
    assert bar.shape[:2] == (31, 301)  # just the bar: rows 40..70, columns 70..370
    assert not panel_visible(bar)
    assert dark_bar(render_panel(None)) is None


def test_result_reads_only_the_text_inside_the_bar():
    seen = []

    def ocr(mask):
        seen.append(mask)
        return "100 metres in 61.3 seconds"

    reader = DistanceReader(ocr=ocr)
    assert reader.result(render_panel("anything")) == RaceResult(100.0, 61.3)
    mask = seen[0]
    assert mask.shape == (31, 301)
    assert 0 < (mask == 0).mean() < 0.3  # white text is ink, the bar is not
    assert reader.result(render_panel(None)) is None


@pytest.mark.skipif(not tesseract_available(), reason="needs tesserocr and English tessdata")
@pytest.mark.parametrize(
    ("text", "expected", "bar_top"),
    [("7.8 metres", RaceResult(7.8, None), 20), ("100 metres in 61.3 seconds", RaceResult(100.0, 61.3), 45)],
)
def test_real_tesseract_reads_both_end_screens(text, expected, bar_top):
    assert DistanceReader().result(render_panel(text, bar_top)) == expected


def test_longest_run():
    assert longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)) == (4, 7)
    assert longest_run(np.array([1, 1], dtype=bool)) == (0, 2)
    assert longest_run(np.zeros(3, dtype=bool)) is None


@pytest.mark.skipif(not tesseract_available(), reason="needs tesserocr and English tessdata")
def test_reads_the_2020_finish_screen():
    # docs/goal.png: the original (2020) agent's finish screen, saved at 500x297.
    from qwop_agent.config import GameConfig

    goal = Path(__file__).parents[1] / "docs" / "goal.png"
    frame = np.asarray(Image.open(goal).convert("RGB").resize((640, 400), Image.Resampling.LANCZOS))
    game = GameConfig()
    x, y, w, h, _, _ = game.result_region.to_pixels(640, 400)
    assert DistanceReader().result(frame[y : y + h, x : x + w]) == RaceResult(100.0, 592.1)
    x, y, w, h, _, _ = game.panel_region.to_pixels(640, 400)
    assert panel_visible(frame[y : y + h, x : x + w])
    x, y, w, h, _, _ = game.score_region.to_pixels(640, 400)
    assert DistanceReader().distance(frame[y : y + h, x : x + w]) == 99.4  # the body was still short of 100 m
