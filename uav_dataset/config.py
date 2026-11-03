"""Shared dataset dimensions, camera assumptions, and output naming rules."""

# -------------------------------------------------------------------------------
# Edit the camera/target values below to match a different aircraft, lens, or
# target size. The background source should show roughly one 200-foot-altitude
# camera footprint; lower-altitude frames zoom into that same source.
# -------------------------------------------------------------------------------

import re

# Sets dimensions for frames in generated dataset
WIDTH = 1280
HEIGHT = 720

# Default camera and target settings
CAMERA_HORIZONTAL_FOV_DEGREES = 82.0
# Altitude changes affect the apparent size of the target in pixels
ALTITUDE_MIN_FEET = 150.0
ALTITUDE_MAX_FEET = 200.0
TARGET_LENGTH_MIN_FEET = 5.0
TARGET_LENGTH_MAX_FEET = 7.0
# Bound the rendered person's long side, while varying the effective camera zoom.
TARGET_MIN_LENGTH_PIXELS = 48.0
TARGET_MAX_LENGTH_PIXELS = 64.0
TARGET_ZOOM_MIN = 1.25
TARGET_ZOOM_MAX = 1.8
# All cutouts in the default object folder depict people.
TARGET_CLASS_NAME = "person"

GENERATED_IMAGE_PATTERN = re.compile(r"^\d{6}\.png$")
CSV_FILENAME = "annotations.csv"
LEGACY_CSV_FILENAME = "labels.csv"
