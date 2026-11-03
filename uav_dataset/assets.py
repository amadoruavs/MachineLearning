"""Input asset models, discovery, validation, and image loading.

Object cutouts must already have an alpha channel; this module does not perform
segmentation. Background photographs are orientation-corrected and cropped by
:func:`prepare_background` to the package's fixed 16:9 output dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import HEIGHT, TARGET_CLASS_NAME, WIDTH

SUPPORTED_SUFFIXES = {suffix.lower() for suffix in Image.registered_extensions()}


@dataclass(frozen=True)
class ObjectAsset:
    """Path and class name for one transparent cutout."""

    path: Path
    class_name: str


def image_paths(directory: Path) -> list[Path]:
    """Recursively list Pillow-supported images in stable case-insensitive order."""
    return sorted(
        (
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: str(path).casefold(),
    )


def load_assets(
    objects_directory: Path, backgrounds_directory: Path
) -> tuple[list[ObjectAsset], list[Path]]:
    """Validate folders and return usable cutout metadata and background paths.

    Args:
        objects_directory: Folder of transparent cutouts, searched recursively.
        backgrounds_directory: Folder of photographs, searched recursively.

    Returns:
        A list of cutout metadata and a list of readable background paths.

    Raises:
        ValueError: If a folder is missing or has no usable assets.
        OSError: If an image file cannot be decoded; the error names the file.
    """
    if not objects_directory.is_dir():
        raise ValueError(f"Object folder does not exist: {objects_directory}")
    if not backgrounds_directory.is_dir():
        raise ValueError(f"Background folder does not exist: {backgrounds_directory}")

    object_assets: list[ObjectAsset] = []
    found_object_images = False
    for path in image_paths(objects_directory):
        found_object_images = True
        try:
            with Image.open(path) as source:
                source.load()
                alpha_min, alpha_max = (
                    source.convert("RGBA").getchannel("A").getextrema()
                )
        except (OSError, ValueError) as error:
            raise OSError(f"Could not read object cutout '{path}': {error}") from error
        if alpha_min < 255 and alpha_max > 0:
            object_assets.append(ObjectAsset(path, TARGET_CLASS_NAME))
    if not object_assets:
        detail = (
            " Found images, but none had usable transparency."
            if found_object_images
            else ""
        )
        raise ValueError(
            f"No transparent object cutouts found in {objects_directory}.{detail}"
        )

    background_paths: list[Path] = []
    for path in image_paths(backgrounds_directory):
        try:
            with Image.open(path) as source:
                source.verify()
        except (OSError, ValueError) as error:
            raise OSError(
                f"Could not read background image '{path}': {error}"
            ) from error
        background_paths.append(path)
    if not background_paths:
        raise ValueError(
            f"No readable background images found in {backgrounds_directory}"
        )
    return object_assets, background_paths


def prepare_background(
    path: Path, rng: np.random.Generator, zoom: float = 1.0
) -> Image.Image:
    """Crop one photograph to 16:9, apply a zoom, and resize it to 1280x720.

    Args:
        path: Background image to load.
        rng: Seeded generator used to choose the crop location.
        zoom: Additional crop fraction in (0, 1]; smaller values zoom in.

    Returns:
        A new RGB Pillow image. Small backgrounds are enlarged with Lanczos.
    """
    if not 0 < zoom <= 1:
        raise ValueError("Background zoom must be greater than 0 and at most 1")
    with Image.open(path) as source:
        background = ImageOps.exif_transpose(source).convert("RGB")
    source_width, source_height = background.size
    target_ratio = WIDTH / HEIGHT
    if source_width / source_height > target_ratio:
        crop_width = max(1, round(source_height * target_ratio))
        available = source_width - crop_width
        left = int(rng.integers(0, available + 1)) if available else 0
        box = (left, 0, left + crop_width, source_height)
    else:
        crop_height = max(1, round(source_width / target_ratio))
        available = source_height - crop_height
        top = int(rng.integers(0, available + 1)) if available else 0
        box = (0, top, source_width, top + crop_height)
    background = background.crop(box)
    crop_width = max(1, round(background.width * zoom))
    crop_height = max(1, round(background.height * zoom))
    max_left = background.width - crop_width
    max_top = background.height - crop_height
    left = int(rng.integers(0, max_left + 1)) if max_left else 0
    top = int(rng.integers(0, max_top + 1)) if max_top else 0
    return background.crop(
        (left, top, left + crop_width, top + crop_height)
    ).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def load_cutout(asset: ObjectAsset) -> Image.Image:
    """Load a cutout as a standalone, orientation-corrected RGBA image."""
    with Image.open(asset.path) as source:
        return ImageOps.exif_transpose(source).convert("RGBA")
