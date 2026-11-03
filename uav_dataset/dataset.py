"""Dataset-generation orchestration and output safety.

This module connects composition, camera simulation, mask-derived annotations,
and file output while leaving each implementation detail in its owning module.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .annotations import visible_bbox
from .assets import ObjectAsset
from .camera import apply_camera_pipeline
from .composition import compose_frame
from .config import CSV_FILENAME, GENERATED_IMAGE_PATTERN, LEGACY_CSV_FILENAME


def _prepare_output(output_directory: Path, overwrite: bool) -> Path:
    """Create the images folder and enforce narrowly scoped overwrite rules.

    With ``overwrite``, only annotation CSVs and six-digit PNGs are removed;
    unrelated files and directories are preserved.
    """
    images_directory = output_directory / "images"
    annotation_paths = (
        output_directory / CSV_FILENAME,
        output_directory / LEGACY_CSV_FILENAME,
    )
    generated = (
        [
            path
            for path in images_directory.iterdir()
            if path.is_file() and GENERATED_IMAGE_PATTERN.fullmatch(path.name)
        ]
        if images_directory.is_dir()
        else []
    )
    has_annotations = any(path.exists() for path in annotation_paths)
    if (has_annotations or generated) and not overwrite:
        raise FileExistsError(
            f"Output already contains generated data in {output_directory}; "
            "choose a new folder or pass --overwrite."
        )
    if overwrite:
        for path in generated:
            path.unlink()
        for path in annotation_paths:
            if path.exists():
                path.unlink()
    images_directory.mkdir(parents=True, exist_ok=True)
    return images_directory


def generate_dataset(
    object_assets: Sequence[ObjectAsset],
    background_paths: Sequence[Path],
    output_directory: Path,
    count: int,
    seed: int,
    overwrite: bool,
    minimum_visible_fraction: float,
) -> None:
    """Generate 720p PNG frames and target-only labels in the output folder.

    Args:
        object_assets: Validated transparent cutouts.
        background_paths: Validated photographs.
        output_directory: Dataset root to create or update.
        count: Positive number of frames to render.
        seed: Seed for scene selection and all augmentation choices.
        overwrite: Replace only prior generator output files when true.
        minimum_visible_fraction: Required in-frame share of target alpha.

    Side effects:
        Writes ``images/NNNNNN.png`` and ``annotations.csv``. Each frame gets
        up to eight placement attempts to meet the in-frame visibility threshold.
    """
    images_directory = _prepare_output(output_directory, overwrite)
    rng = np.random.default_rng(seed)
    annotations_path = output_directory / CSV_FILENAME
    with annotations_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("filename", "class", "x", "y", "w", "h"))
        for index in range(1, count + 1):
            target = object_assets[int(rng.integers(len(object_assets)))]
            for _ in range(8):
                image, mask, fraction = compose_frame(
                    target, background_paths, rng
                )
                if fraction >= minimum_visible_fraction:
                    break

            rendered, final_mask = apply_camera_pipeline(image, mask, rng)
            box = visible_bbox(final_mask)
            if box is None:
                raise RuntimeError(
                    f"Target disappeared from frame {index}; check cutout alpha."
                )
            name = f"{index:06d}.png"
            Image.fromarray(rendered, mode="RGB").save(images_directory / name)
            writer.writerow((f"images/{name}", target.class_name, *box))
            if index == 1 or index == count or index % 25 == 0:
                print(f"Generated {index}/{count} frames", file=sys.stderr)
