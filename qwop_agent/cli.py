"""Command line: ``qwop-agent train | evaluate | compare | random | plot``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .config import EnvConfig, GameConfig


def _env_config(args: argparse.Namespace) -> EnvConfig:
    game = GameConfig(
        headless=not getattr(args, "headed", False),
        realtime=getattr(args, "realtime", False),
    )
    return EnvConfig(game=game)


def _add_view_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--headed", action="store_true", help="show the browser window instead of running headless")
    parser.add_argument("--realtime", action="store_true", help="play at normal speed (default: as fast as possible)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qwop-agent",
        description="Teach a PPO agent to run 100 m in the official browser version of QWOP.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train (or resume) a PPO agent")
    train.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/qwop-ppo"),
        help="where checkpoints and logs go (default: %(default)s)",
    )
    train.add_argument("--n-envs", type=int, default=8, help="browsers playing in parallel (default: %(default)s)")
    train.add_argument(
        "--timesteps", type=int, default=20_000_000, help="agent steps to train for (default: %(default)s)"
    )
    train.add_argument("--max-hours", type=float, default=None, help="stop after this much wall-clock time")
    train.add_argument(
        "--checkpoint-every", type=int, default=100_000, help="save a checkpoint every N steps (default: %(default)s)"
    )
    train.add_argument(
        "--stop-after-finishes",
        type=int,
        default=3,
        help="stop once the agent has finished 100 m this many times (default: %(default)s)",
    )
    train.add_argument("--resume", type=Path, default=None, help="continue from a checkpoint .zip")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto", help="torch device: auto, cuda or cpu (default: %(default)s)")

    evaluate = commands.add_parser("evaluate", help="let a trained agent run and optionally record a GIF")
    evaluate.add_argument("model", type=Path, help="a checkpoint .zip from training, or an exported policy")
    evaluate.add_argument("--episodes", type=int, default=1, help="races to run (default: %(default)s)")
    evaluate.add_argument(
        "--stochastic", action="store_true", help="sample actions instead of taking the most likely one"
    )
    evaluate.add_argument("--gif", type=Path, default=None, help="save the best race as a GIF here")
    evaluate.add_argument("--label", default=None, help="caption drawn on the GIF")
    evaluate.add_argument(
        "--max-gif-mb", type=float, default=5.0, help="shrink the GIF until it fits (default: %(default)s)"
    )
    evaluate.add_argument("--gif-speed", type=float, default=1.0, help="playback speed of the GIF, e.g. 4 for 4x")
    evaluate.add_argument(
        "--export-policy", type=Path, default=None, help="also save just the policy weights (small file) here"
    )
    _add_view_options(evaluate)

    compare = commands.add_parser("compare", help="side-by-side GIF of two agents, e.g. early vs trained")
    compare.add_argument("left", type=Path, help="model shown on the left")
    compare.add_argument("right", type=Path, help="model shown on the right")
    compare.add_argument("--labels", nargs=2, default=("early training", "trained"), metavar=("LEFT", "RIGHT"))
    compare.add_argument("--gif", type=Path, required=True, help="output GIF")
    compare.add_argument("--seconds", type=float, default=20.0, help="game seconds to show (default: %(default)s)")
    compare.add_argument("--max-gif-mb", type=float, default=5.0)

    random_play = commands.add_parser("random", help="press random keys to check that the game environment works")
    random_play.add_argument("--episodes", type=int, default=3, help="races to run (default: %(default)s)")
    random_play.add_argument("--seed", type=int, default=0)
    _add_view_options(random_play)

    plot = commands.add_parser("plot", help="draw the learning curve of a run")
    plot.add_argument("run_dir", type=Path, help="the --run-dir of a training run")
    plot.add_argument(
        "--out", type=Path, default=Path("docs/images/learning-curve.png"), help="output image (default: %(default)s)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.command == "train":
        from .train import TrainSettings, train

        settings = TrainSettings(
            run_dir=args.run_dir,
            n_envs=args.n_envs,
            total_timesteps=args.timesteps,
            max_hours=args.max_hours,
            checkpoint_every=args.checkpoint_every,
            stop_after_finishes=args.stop_after_finishes,
            resume=args.resume,
            seed=args.seed,
            device=args.device,
        )
        final = train(settings, _env_config(args))
        print(f"saved {final}")
    elif args.command == "evaluate":
        from .evaluate import evaluate

        evaluate(
            args.model,
            episodes=args.episodes,
            deterministic=not args.stochastic,
            gif=args.gif,
            label=args.label,
            max_gif_bytes=int(args.max_gif_mb * 1_000_000),
            gif_speed=args.gif_speed,
            export_policy=args.export_policy,
            env_config=_env_config(args),
        )
    elif args.command == "compare":
        from .evaluate import compare

        config = _env_config(args)
        max_steps = round(args.seconds / config.step_seconds)
        compare(args.left, args.right, args.labels, args.gif, max_steps, int(args.max_gif_mb * 1_000_000), config)
    elif args.command == "random":
        from .evaluate import play_random

        play_random(args.episodes, seed=args.seed, env_config=_env_config(args))
    elif args.command == "plot":
        from .plot import plot_learning_curve

        out = plot_learning_curve(args.run_dir, args.out)
        print(f"saved {out}")
    return 0


__all__ = ["build_parser", "main"]
