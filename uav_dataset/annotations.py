"""Target-mask annotation utilities.

Masks are floating alpha coverage maps aligned to the final rendered image.
Boxes use clipped, top-left-origin pixel coordinates and intentionally ignore
faint interpolation fringes after camera warps.
"""

import numpy as np


def visible_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return visible target ``(x, y, width, height)`` or ``None``.

    Mask values at or below 0.15 are ignored so bilinear warp fringes do not
    inflate a box. Since the mask is frame-sized, returned coordinates are
    already clipped to the image bounds.
    """
    rows, columns = np.nonzero(mask > 0.15)
    if rows.size == 0:
        return None
    left, top = int(columns.min()), int(rows.min())
    return left, top, int(columns.max()) - left + 1, int(rows.max()) - top + 1
