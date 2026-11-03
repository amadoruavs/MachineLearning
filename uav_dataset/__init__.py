"""Generate synthetic 1280x720 UAV person-detection datasets.

Run the package with ``python -m uav_dataset --help``. The package composites
one transparent person cutout onto a background, scales it for
the configured altitude and camera field of view, and writes PNG frames with
target-only pixel bounding boxes. See `uav_dataset.config` for the
editable camera and target assumptions.
"""
