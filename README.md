# Dataset Generator

Generate synthetic 1280x720 images for person-detection experiments. The
generator places transparent person cutouts over background photographs,
applies approximate camera effects, and writes pixel bounding-box annotations.

## Requirements and setup

Python 3.10 or newer is required.

The required Python libraries are **NumPy** (`numpy`) and **Pillow** (`Pillow`).
Installing from `requirements.txt` selects the latest releases compatible with
your Python version at install time.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run

From the repository root, generate 20 frames into a new output directory:

```bash
python -m uav_dataset --count 20 --output out_generated
```

By default, the command reads transparent cutouts from `objects/` and
background images from `backgrounds/`. The output contains PNGs in
`images/` and an `annotations.csv` file with each image's class and bounding
box. Use `python -m uav_dataset --help` to see all options. Output paths are
relative to the current working directory. Use a new output directory for
each run, or pass `--overwrite` to replace generated files in an existing one.

## Repository layout

- `uav_dataset/` — generator package and configurable camera/target settings.
- `objects/` — transparent cutouts used as detection targets.
- `backgrounds/` — background images used for composition.
- `source_photos/` — source and cropped images used to prepare some assets.
