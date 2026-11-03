"""Command-line parsing and user-facing validation for ``uav_dataset``.

The CLI owns argument defaults and turns known input/output errors into concise
argparse messages. It delegates asset work and generation to their modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .assets import load_assets
from .dataset import generate_dataset


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without filesystem side effects."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate 1280x720 UAV person-detection images with one target per "
            "frame. Edit uav_dataset/config.py to change camera assumptions."
        )
    )
    parser.add_argument(
        "--objects",
        type=Path,
        default=Path("objects"),
        help="cutout folder (default: ./objects)",
    )
    parser.add_argument(
        "--backgrounds",
        type=Path,
        default=Path("backgrounds"),
        help="background folder (default: ./backgrounds)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out"),
        help="dataset folder (default: ./out)",
    )
    parser.add_argument(
        "--count", type=int, default=20, help="number of frames (default: 20)"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--min-visible-fraction",
        type=float,
        default=0.45,
        help="minimum visible target fraction before retry (default: 0.45)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace prior six-digit PNGs and annotation CSVs",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse options, validate inputs, and run dataset generation."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    if arguments.count <= 0:
        parser.error("--count must be greater than zero")
    if not 0 < arguments.min_visible_fraction <= 1:
        parser.error("--min-visible-fraction must be greater than 0 and at most 1")

    try:
        objects, backgrounds = load_assets(arguments.objects, arguments.backgrounds)
        generate_dataset(
            objects,
            backgrounds,
            arguments.output,
            arguments.count,
            arguments.seed,
            arguments.overwrite,
            arguments.min_visible_fraction,
        )
    except (FileExistsError, OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0
