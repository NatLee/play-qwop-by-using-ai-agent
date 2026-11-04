"""Draw a run's learning curve from its ``episodes-*.monitor.csv`` files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Chart colours (a validated two-series palette on a light surface).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MEAN_COLOR = "#2a78d6"
BEST_COLOR = "#eb6834"


@dataclass(frozen=True)
class Episode:
    step: int  # training steps done when the race ended
    hours: float  # wall-clock hours since training started
    distance: float
    finished: bool


def load_episodes(run_dir: Path) -> list[Episode]:
    """Episodes of every segment of a run, in training order."""
    files = sorted(run_dir.glob("episodes-*.monitor.csv"), key=lambda p: int(p.name.split("-")[1].split(".")[0]))
    episodes: list[Episode] = []
    steps, hours_offset = 0, 0.0
    for path in files:
        with path.open(encoding="utf-8") as handle:
            handle.readline()  # "#{...}" metadata line written by SB3
            rows = sorted(csv.DictReader(handle), key=lambda row: float(row["t"]))
        last_hours = 0.0
        for row in rows:
            steps += int(row["l"])
            last_hours = float(row["t"]) / 3600
            episodes.append(
                Episode(steps, hours_offset + last_hours, float(row["distance"]), row["finished"] == "True")
            )
        hours_offset += last_hours
    return episodes


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` values (of all values so far at the start)."""
    sums = np.cumsum(np.insert(np.asarray(values, dtype=float), 0, 0.0))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - window, 0)
    return (sums[ends] - sums[starts]) / (ends - starts)


def plot_learning_curve(run_dir: Path, out: Path, window: int = 20) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = load_episodes(run_dir)
    if not episodes:
        raise ValueError(f"no episodes found in {run_dir}")
    # Wall-clock time on the x axis: it is exact, and it is what "how long
    # did it take" means. Steps go in the subtitle.
    hours = np.array([e.hours for e in episodes])
    scale, unit = (1.0, "hours") if hours[-1] >= 2 else (60.0, "minutes")
    when = hours * scale
    distance = np.array([e.distance for e in episodes])
    mean = rolling_mean(distance, window)
    best = np.maximum.accumulate(distance)
    finishes = sum(e.finished for e in episodes)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(when, distance, s=12, color=MUTED, alpha=0.4, linewidths=0, label="one race", zorder=1)
    ax.plot(
        when,
        mean,
        color=MEAN_COLOR,
        linewidth=2,
        solid_capstyle="round",
        label=f"average of last {window} races",
        zorder=3,
    )
    ax.step(when, best, where="post", color=BEST_COLOR, linewidth=2, label="best so far", zorder=2)
    for values, color in ((mean, MEAN_COLOR), (best, BEST_COLOR)):
        ax.annotate(
            f"{values[-1]:.1f} m",
            (when[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=10,
        )
        ax.plot(
            when[-1], values[-1], "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4
        )

    ax.set_xlabel(f"training time ({unit})", color=INK_SECONDARY)
    ax.set_ylabel("distance per race (m)", color=INK_SECONDARY)
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED)
    low = min(0.0, float(np.percentile(distance, 1)))
    ax.set_ylim(low, max(best[-1] * 1.12, 5.0))
    ax.set_xlim(0, when[-1] * 1.08)
    fig.suptitle("How far the agent gets in QWOP while it learns", x=0.07, ha="left", fontsize=14, color=INK)
    ax.set_title(
        f"PPO on {len(episodes):,} races ({episodes[-1].step:,} steps), "
        f"{when[-1]:.0f} {unit} of training, finished 100 m: {finishes}",
        loc="left",
        fontsize=10,
        color=INK_SECONDARY,
    )
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10, markerscale=3)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
