"""The original entry point, kept so the old commands still work.

    python main.py                                  -> train, then run one race
    python main.py --train False --model_path M     -> run one race with model M
    python main.py --test_env True                  -> press random keys
    --model_save_path P                             -> also copy the trained model to P
    --chrome_driver_path                            -> accepted and ignored

The ``qwop-agent`` command (see README) has more options. This file only maps
the old flags onto it and can be removed once nobody uses them.
"""

from __future__ import annotations

import argparse
import ast
import logging
import shutil
from pathlib import Path

DEFAULT_MODEL = Path(__file__).with_name("model") / "qwop-ppo.zip"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play QWOP with a reinforcement learning agent (legacy flags).")
    parser.add_argument("--train", type=ast.literal_eval, default=True, help='Train the model: "True" or "False".')
    parser.add_argument("--model_path", type=Path, default=None, help="Model to load (a .zip saved by training).")
    parser.add_argument("--model_save_path", type=Path, default=None, help="Copy the trained model here.")
    parser.add_argument(
        "--test_env", type=ast.literal_eval, default=False, help='Press random keys: "True" or "False".'
    )
    parser.add_argument(
        "--chrome_driver_path", type=str, default=None, help="Ignored: Playwright manages the browser now."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M")
    if args.chrome_driver_path:
        logging.warning("--chrome_driver_path is no longer needed and is ignored (Playwright brings its own browser)")

    if args.test_env:
        from qwop_agent.evaluate import play_random

        play_random(episodes=None)
        return 0

    from qwop_agent.evaluate import evaluate

    model = args.model_path
    if args.train:
        from qwop_agent.train import TrainSettings, train

        model = train(TrainSettings(resume=args.model_path))
        if args.model_save_path is not None:
            args.model_save_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(model, args.model_save_path)
            logging.info("model saved to %s", args.model_save_path)
    races = evaluate(model or DEFAULT_MODEL)
    logging.info("Total Episode Reward: %.2f", races[0].reward)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
