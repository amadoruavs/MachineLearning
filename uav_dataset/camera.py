"""Approximate camera geometry and sensor effects for each composite frame.

The effects are designed for visual variation, not as a physically exact
camera model. Geometric transformations are applied to the target mask using the
same inverse coordinate maps as the image so pixel boxes remain aligned.
"""

from __future__ import annotations

import math

import numpy as np

from .config import WIDTH


def _bilinear_sample(
    array: np.ndarray, source_x: np.ndarray, source_y: np.ndarray
) -> np.ndarray:
    """Sample a 2D or channel-last image at floating coordinates bilinearly.

    Coordinates outside the image are clamped to the closest edge pixel to
    avoid black wedges from the small camera warps used here.
    """
    height, width = array.shape[:2]
    x = np.clip(source_x, 0, width - 1)
    y = np.clip(source_y, 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
    if array.ndim == 3:
        wx, wy = wx[..., None], wy[..., None]
    top = (
        array[y0, x0].astype(np.float32) * (1 - wx)
        + array[y0, x1].astype(np.float32) * wx
    )
    bottom = (
        array[y1, x0].astype(np.float32) * (1 - wx)
        + array[y1, x1].astype(np.float32) * wx
    )
    return top * (1 - wy) + bottom * wy


def _motion_blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply mild linear whole-frame motion blur using five shutter samples."""
    distance = float(rng.uniform(0.0, 5.0))
    if distance < 0.5:
        return image
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    dx, dy = math.cos(angle) * distance, math.sin(angle) * distance
    height, width = image.shape[:2]
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    output = np.zeros_like(image, dtype=np.float32)
    for offset in np.linspace(-0.5, 0.5, 5):
        output += _bilinear_sample(image, x - dx * offset, y - dy * offset)
    return output / 5.0


def _rolling_shutter(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a small row-dependent horizontal displacement to image and mask."""
    height, width = mask.shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    shift = float(rng.uniform(-0.012, 0.012) * WIDTH)
    source_x = x - shift * (y / max(1, height - 1) - 0.5)
    return _bilinear_sample(image, source_x, y), _bilinear_sample(mask, source_x, y)


def _lens_distortion(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply mild inverse Brown-Conrady radial and tangential distortion.

    Five fixed-point iterations invert the sampled forward distortion model.
    Using one inverse map for image and mask keeps annotations geometrically
    aligned with the generated frame.
    """
    height, width = mask.shape
    focal = max(width, height) * 0.5
    center_x, center_y = (width - 1) * 0.5, (height - 1) * 0.5
    x, y = np.meshgrid(
        (np.arange(width, dtype=np.float32) - center_x) / focal,
        (np.arange(height, dtype=np.float32) - center_y) / focal,
    )
    k1, k2 = float(rng.uniform(-0.035, 0.035)), float(rng.uniform(-0.008, 0.008))
    p1, p2 = float(rng.uniform(-0.0012, 0.0012)), float(rng.uniform(-0.0012, 0.0012))
    source_x, source_y = x.copy(), y.copy()
    for _ in range(5):
        radius2 = source_x * source_x + source_y * source_y
        radial = 1.0 + k1 * radius2 + k2 * radius2**2
        tangent_x = 2 * p1 * source_x * source_y + p2 * (radius2 + 2 * source_x**2)
        tangent_y = p1 * (radius2 + 2 * source_y**2) + 2 * p2 * source_x * source_y
        source_x = (x - tangent_x) / np.maximum(radial, 0.5)
        source_y = (y - tangent_y) / np.maximum(radial, 0.5)
    pixel_x, pixel_y = source_x * focal + center_x, source_y * focal + center_y
    return (
        _bilinear_sample(image, pixel_x, pixel_y),
        _bilinear_sample(mask, pixel_x, pixel_y),
    )


def _bayer_demosaic(raw: np.ndarray) -> np.ndarray:
    """Reconstruct RGB from an RGGB mosaic using normalized bilinear filters."""
    height, width = raw.shape
    if height % 2 or width % 2:
        raise ValueError("Bayer demosaicing requires even image dimensions")
    red, green, blue = (np.zeros_like(raw, dtype=np.float32) for _ in range(3))
    red[0::2, 0::2] = 1.0
    green[0::2, 1::2] = green[1::2, 0::2] = 1.0
    blue[1::2, 1::2] = 1.0
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)

    def interpolate(sites: np.ndarray) -> np.ndarray:
        """Normalized-convolve one color plane from its measured CFA sites."""
        samples = np.pad(raw * sites, 1, mode="constant")
        weights = np.pad(sites, 1, mode="constant")
        numerator, denominator = np.zeros_like(raw), np.zeros_like(raw)
        for row in range(3):
            for column in range(3):
                kernel_weight = kernel[row, column]
                numerator += (
                    samples[row : row + height, column : column + width] * kernel_weight
                )
                denominator += (
                    weights[row : row + height, column : column + width] * kernel_weight
                )
        return np.divide(
            numerator, denominator, out=np.zeros_like(raw), where=denominator > 0
        )

    return np.stack((interpolate(red), interpolate(green), interpolate(blue)), axis=2)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Encode clipped linear-light values as display-referred sRGB."""
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def apply_camera_pipeline(
    image: np.ndarray, target_mask: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply geometric camera and RGB sensor effects to one composite.

    Order: motion blur, rolling shutter, lens distortion, linear exposure and
    white balance, illumination gradient, shot/read noise, RGGB sampling,
    bilinear demosaicing, vignetting, and sRGB encoding. Only geometric stages
    modify the target mask.

    Args:
        image: Composited uint8 RGB frame.
        target_mask: Aligned float mask in [0, 1].
        rng: Seeded generator shared with the dataset.

    Returns:
        Processed uint8 RGB frame and geometrically transformed float mask.
    """
    processed = _motion_blur(image.astype(np.float32), rng)
    processed, mask = _rolling_shutter(processed, target_mask, rng)
    processed, mask = _lens_distortion(processed, mask, rng)
    srgb = np.clip(processed / 255.0, 0.0, 1.0)
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    exposure = float(2.0 ** rng.uniform(-2.2, 0.8))
    white_balance = rng.uniform(0.88, 1.12, size=(1, 1, 3)).astype(np.float32)
    scene = np.clip(linear * exposure * white_balance, 0.0, 1.0)

    height, width = scene.shape[:2]
    x, y = np.meshgrid(
        np.linspace(-1, 1, width, dtype=np.float32),
        np.linspace(-1, 1, height, dtype=np.float32),
    )
    direction = float(rng.uniform(0, 2 * math.pi))
    gradient = x * math.cos(direction) + y * math.sin(direction)
    scene *= 1.0 + gradient[..., None] * float(rng.uniform(-0.075, 0.075))
    capacity = float(rng.uniform(3000, 10000))
    noisy = rng.poisson(np.clip(scene, 0, 1) * capacity).astype(np.float32) / capacity
    noisy += rng.normal(0, float(rng.uniform(0.0015, 0.009)), size=noisy.shape).astype(
        np.float32
    )
    noisy = np.clip(noisy, 0, 1)

    raw = np.zeros((height, width), dtype=np.float32)
    raw[0::2, 0::2] = noisy[0::2, 0::2, 0]
    raw[0::2, 1::2] = noisy[0::2, 1::2, 1]
    raw[1::2, 0::2] = noisy[1::2, 0::2, 1]
    raw[1::2, 1::2] = noisy[1::2, 1::2, 2]
    rgb = _bayer_demosaic(raw)
    radius2 = x * x + y * y
    vignette = np.clip(1.0 - float(rng.uniform(0.02, 0.16)) * radius2, 0.72, 1.0)
    output = np.uint8(
        np.clip(np.round(_linear_to_srgb(rgb * vignette[..., None]) * 255), 0, 255)
    )
    return output, np.clip(mask, 0, 1)
