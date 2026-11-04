"""Write race recordings as GIFs that stay under a size limit."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# (scale of the source width, keep every n-th frame), best first. A readable
# picture matters more than a smooth one, so frames are dropped before the
# picture shrinks much. The first combination that fits the size limit wins.
QUALITY_STEPS: tuple[tuple[float, int], ...] = (
    (0.75, 1),
    (0.75, 2),
    (0.75, 3),
    (0.625, 3),
    (0.625, 4),
    (0.5, 4),
    (0.5, 6),
    (0.4, 8),
    (0.3, 12),
    (0.25, 16),
)
PALETTE_COLORS = 96
# Browsers show GIF frames shorter than this at a crawl, so never go below it.
MIN_FRAME_MS = 20


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    """Return ``frame`` with ``text`` on a dark bar along the bottom."""
    image = Image.fromarray(frame)
    height = max(18, image.height // 14)
    font = ImageFont.load_default(size=int(height * 0.7))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, image.height - height, image.width, image.height), fill=(20, 24, 32))
    draw.text((8, image.height - height / 2), text, fill=(255, 255, 255), font=font, anchor="lm")
    return np.asarray(image)


def side_by_side(left: Sequence[np.ndarray], right: Sequence[np.ndarray], gap: int = 6) -> list[np.ndarray]:
    """Pair two recordings frame by frame; the shorter one holds its last frame."""
    frames = []
    for i in range(max(len(left), len(right))):
        a = left[min(i, len(left) - 1)]
        b = right[min(i, len(right) - 1)]
        spacer = np.full((a.shape[0], gap, 3), 255, dtype=np.uint8)
        frames.append(np.concatenate([a, spacer, b], axis=1))
    return frames


def encode_gif(frames: Sequence[np.ndarray], width: int, frame_ms: int) -> bytes:
    """Encode frames at ``width`` pixels wide with one shared palette."""
    if not frames:
        raise ValueError("no frames to encode")
    first = Image.fromarray(frames[0])
    height = max(1, round(first.height * width / first.width))
    resized = [Image.fromarray(f).resize((width, height), Image.Resampling.LANCZOS) for f in frames]
    # One palette for the whole clip avoids colour flicker between frames.
    samples = resized[:: max(1, len(resized) // 12)]
    mosaic = Image.new("RGB", (width, height * len(samples)))
    for i, image in enumerate(samples):
        mosaic.paste(image, (0, i * height))
    palette = mosaic.quantize(colors=PALETTE_COLORS, method=Image.Quantize.MEDIANCUT)
    paletted = [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in resized]
    buffer = io.BytesIO()
    paletted[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=paletted[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return buffer.getvalue()


def write_gif(
    frames: Sequence[np.ndarray], path: Path, frame_seconds: float, max_bytes: int, speed: float = 1.0
) -> tuple[int, int]:
    """Save the best-looking GIF that fits ``max_bytes``; return (width, frame stride).

    ``frame_seconds`` is the game time between frames; ``speed`` > 1 makes a time-lapse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    source_width = frames[0].shape[1]
    for scale, stride in QUALITY_STEPS:
        width = round(source_width * scale)
        kept = list(frames[::stride])
        if len(frames) > 1 and (len(frames) - 1) % stride:
            kept.append(frames[-1])  # always show how the race ended
        data = encode_gif(kept, width, max(MIN_FRAME_MS, round(frame_seconds * stride * 1000 / speed)))
        if len(data) <= max_bytes:
            path.write_bytes(data)
            return width, stride
    raise ValueError(f"cannot fit {len(frames)} frames into {max_bytes} bytes")
