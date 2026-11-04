import numpy as np
import pytest
from PIL import Image

from qwop_agent.gif import QUALITY_STEPS, caption, side_by_side, write_gif


def noisy_frames(n=40, size=(200, 320)):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, size=(*size, 3), dtype=np.uint8) for _ in range(n)]


def test_caption_keeps_the_frame_size():
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    labelled = caption(frame, "trained")
    assert labelled.shape == frame.shape
    assert labelled[-2, 2].tolist() != [0, 0, 0]  # the caption bar is drawn


def test_side_by_side_holds_the_last_frame_of_the_shorter_clip():
    left = [np.full((4, 5, 3), i, dtype=np.uint8) for i in range(3)]
    right = [np.full((4, 5, 3), 100 + i, dtype=np.uint8) for i in range(5)]
    frames = side_by_side(left, right, gap=2)
    assert len(frames) == 5
    assert frames[0].shape == (4, 12, 3)
    assert frames[4][0, 0, 0] == 2  # left clip stays on its last frame
    assert frames[4][0, -1, 0] == 104


def test_gif_shrinks_until_it_fits(tmp_path):
    path = tmp_path / "race.gif"
    frames = noisy_frames()
    width, stride = write_gif(frames, path, frame_seconds=0.1, max_bytes=300_000)
    assert path.stat().st_size <= 300_000
    assert (width, stride) in [(round(320 * scale), s) for scale, s in QUALITY_STEPS]
    with Image.open(path) as gif:
        assert gif.width == width
        assert gif.n_frames == len(frames[::stride]) + (1 if (len(frames) - 1) % stride else 0)
        assert gif.info["duration"] == pytest.approx(100 * stride, abs=10)


def test_gif_that_cannot_fit_raises(tmp_path):
    with pytest.raises(ValueError):
        write_gif(noisy_frames(n=4), tmp_path / "x.gif", frame_seconds=0.1, max_bytes=10)


def test_time_lapse_shortens_frames_but_not_below_the_browser_minimum(tmp_path):
    frames = noisy_frames(n=6, size=(20, 32))
    fast = tmp_path / "fast.gif"
    write_gif(frames, fast, frame_seconds=0.1, max_bytes=10_000_000, speed=4)
    with Image.open(fast) as gif:
        assert gif.info["duration"] == pytest.approx(25, abs=5)
    write_gif(frames, fast, frame_seconds=0.1, max_bytes=10_000_000, speed=100)
    with Image.open(fast) as gif:
        assert gif.info["duration"] >= 20
