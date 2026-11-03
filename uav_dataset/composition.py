"""Altitude-scaled scene composition from cutouts and background photographs.

This module owns target transforms, frame clipping, contact-shadow drawing,
and alpha compositing. Target pixel size and background zoom are tied to the
same sampled altitude and configured camera field of view.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .assets import ObjectAsset, load_cutout, prepare_background
from .config import (
    ALTITUDE_MAX_FEET,
    ALTITUDE_MIN_FEET,
    CAMERA_HORIZONTAL_FOV_DEGREES,
    HEIGHT,
    TARGET_MAX_LENGTH_PIXELS,
    TARGET_MIN_LENGTH_PIXELS,
    TARGET_LENGTH_MAX_FEET,
    TARGET_LENGTH_MIN_FEET,
    TARGET_ZOOM_MAX,
    TARGET_ZOOM_MIN,
    WIDTH,
)


def _transform_cutout(
    asset: ObjectAsset,
    rng: np.random.Generator,
    target_length_pixels: float,
) -> Image.Image:
    """Scale a cropped cutout to its altitude-derived pixel length and rotate it.

    Args:
        asset: Transparent cutout to transform.
        rng: Seeded generator for the orientation choice.
        target_length_pixels: Expected visible long dimension in output pixels.

    Returns:
        Transformed RGBA sprite with its rotated extent retained. Its unrotated
        long side matches the projected length configured for the altitude.
    """
    image = load_cutout(asset)
    alpha_bounds = image.getchannel("A").getbbox()
    if alpha_bounds is None:
        raise ValueError(f"Object cutout has no visible pixels: {asset.path}")
    image = image.crop(alpha_bounds)
    original_width, original_height = image.size
    scale = target_length_pixels / max(original_width, original_height)
    scaled = image.resize(
        (
            max(1, round(original_width * scale)),
            max(1, round(original_height * scale)),
        ),
        Image.Resampling.LANCZOS,
    )
    if rng.random() < 0.5:
        scaled = ImageOps.mirror(scaled)
    return scaled.rotate(
        float(rng.uniform(-32.0, 32.0)),
        resample=Image.Resampling.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )


def _sprite_patch(
    sprite: Image.Image, left: int, top: int
) -> tuple[Image.Image, tuple[int, int]] | None:
    """Return the in-frame part of a sprite and its destination coordinate."""
    right, bottom = left + sprite.width, top + sprite.height
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(WIDTH, right), min(HEIGHT, bottom)
    if x0 >= x1 or y0 >= y1:
        return None
    crop = sprite.crop((x0 - left, y0 - top, x1 - left, y1 - top))
    return crop, (x0, y0)


def _alpha_on_canvas(alpha: Image.Image, left: int, top: int) -> np.ndarray:
    """Place sprite alpha into a frame-sized float mask, clipping at edges."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    patch = _sprite_patch(alpha, left, top)
    if patch is None:
        return mask
    sprite_patch, (x0, y0) = patch
    values = np.asarray(sprite_patch, dtype=np.float32) / 255.0
    mask[y0 : y0 + values.shape[0], x0 : x0 + values.shape[1]] = values
    return mask


def _paste_sprite(
    canvas: Image.Image, sprite: Image.Image, left: int, top: int
) -> None:
    """Alpha-composite a sprite onto an RGBA canvas, safely clipping its edges."""
    patch = _sprite_patch(sprite, left, top)
    if patch is not None:
        cropped, position = patch
        canvas.alpha_composite(cropped, position)


def _add_contact_shadow(
    scene: Image.Image,
    target_alpha: np.ndarray,
    rng: np.random.Generator,
    target_size: float,
) -> None:
    """Add a subdued, offset, blurred 2D shadow under the target silhouette.

    It has no ground plane, depth, or sun visibility; it is an image-space
    contact cue rather than a physical shadow calculation.
    """
    alpha_image = Image.fromarray(np.uint8(np.clip(target_alpha * 255.0, 0, 255)))
    offset_x = int(rng.uniform(-0.025, 0.025) * target_size)
    offset_y = max(1, int(rng.uniform(0.015, 0.055) * target_size))
    shifted = Image.new("L", (WIDTH, HEIGHT), 0)
    shifted.paste(alpha_image, (offset_x, offset_y))
    blur_radius = max(1.0, target_size * float(rng.uniform(0.018, 0.045)))
    blurred = shifted.filter(ImageFilter.GaussianBlur(blur_radius))
    shadow_alpha = np.asarray(blurred, dtype=np.float32) * float(
        rng.uniform(0.10, 0.30)
    )
    shadow = Image.new("RGBA", (WIDTH, HEIGHT), (18, 22, 20, 0))
    shadow.putalpha(Image.fromarray(np.uint8(np.clip(shadow_alpha, 0, 255))))
    scene.alpha_composite(shadow)


def compose_frame(
    target_asset: ObjectAsset,
    background_paths: Sequence[Path],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Composite one target and its alpha mask over a background.

    Args:
        target_asset: The one cutout labeled in this frame.
        background_paths: Validated background photographs.
        rng: Shared seeded generator for reproducible choices.

    Returns:
        Uint8 RGB scene, float target mask, and fraction of target alpha that
        lies in frame. The background zoom assumes its uncropped image
        represents the camera footprint at ``ALTITUDE_MAX_FEET``.
    """
    if not 0 < ALTITUDE_MIN_FEET < ALTITUDE_MAX_FEET:
        raise ValueError("Altitude settings must satisfy 0 < minimum < maximum")
    if not 0 < CAMERA_HORIZONTAL_FOV_DEGREES < 180:
        raise ValueError("Camera horizontal field of view must be between 0 and 180")
    if not 0 < TARGET_LENGTH_MIN_FEET <= TARGET_LENGTH_MAX_FEET:
        raise ValueError("Target length bounds must satisfy 0 < minimum <= maximum")
    if not 0 < TARGET_MIN_LENGTH_PIXELS <= TARGET_MAX_LENGTH_PIXELS:
        raise ValueError("Target pixel bounds must satisfy 0 < minimum <= maximum")
    if not 0 < TARGET_ZOOM_MIN <= TARGET_ZOOM_MAX:
        raise ValueError("Target zoom bounds must satisfy 0 < minimum <= maximum")

    altitude_feet = float(rng.uniform(ALTITUDE_MIN_FEET, ALTITUDE_MAX_FEET))
    altitude_meters = altitude_feet * 0.3048
    target_length_feet = float(
        rng.uniform(TARGET_LENGTH_MIN_FEET, TARGET_LENGTH_MAX_FEET)
    )
    target_length_meters = target_length_feet * 0.3048
    horizontal_fov_radians = math.radians(CAMERA_HORIZONTAL_FOV_DEGREES)
    ground_width_meters = (
        2.0 * altitude_meters * math.tan(horizontal_fov_radians / 2.0)
    )
    pixels_per_meter = WIDTH / ground_width_meters
    projected_length_pixels = target_length_meters * pixels_per_meter
    zoom_factor = float(rng.uniform(TARGET_ZOOM_MIN, TARGET_ZOOM_MAX))
    target_length_pixels = min(
        max(
            projected_length_pixels * zoom_factor,
            TARGET_MIN_LENGTH_PIXELS,
        ),
        TARGET_MAX_LENGTH_PIXELS,
    )
    effective_zoom = target_length_pixels / projected_length_pixels
    background_zoom = (altitude_feet / ALTITUDE_MAX_FEET) / effective_zoom

    background_path = background_paths[int(rng.integers(len(background_paths)))]
    scene = prepare_background(background_path, rng, background_zoom).convert("RGBA")
    target = _transform_cutout(target_asset, rng, target_length_pixels)
    target_size = float(max(target.size))
    left = int(rng.uniform(0.08, 0.92) * WIDTH - target.width / 2)
    top = int(rng.uniform(0.10, 0.92) * HEIGHT - target.height / 2)
    target_alpha = _alpha_on_canvas(target.getchannel("A"), left, top)
    target_area = float(np.asarray(target.getchannel("A"), dtype=np.float32).sum())
    _add_contact_shadow(scene, target_alpha, rng, target_size)
    _paste_sprite(scene, target, left, top)

    in_frame_fraction = (
        float(np.sum(target_alpha) * 255.0) / target_area if target_area else 0.0
    )
    return np.asarray(scene.convert("RGB")), target_alpha, in_frame_fraction
