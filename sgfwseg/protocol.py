"""Dataset presets and shared patch input checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from .io import load_image, load_label

CLASS_NAMES = json.loads(Path(__file__).with_name("classes.json").read_text(encoding="utf-8"))

RGB_CLASSES = {
    "gid15": (
        (150, 250, 0), (250, 0, 150), (150, 0, 250), (200, 0, 0), (0, 0, 200),
        (200, 150, 150), (250, 150, 150), (150, 200, 150), (0, 200, 0), (0, 150, 200),
        (250, 200, 0), (0, 200, 250), (200, 0, 200), (150, 150, 250), (200, 200, 0),
    ),
    "hiertoulouse": (
        (255, 0, 128), (255, 128, 128), (255, 255, 128), (0, 255, 255), (128, 255, 0),
        (0, 128, 0), (128, 128, 0), (128, 255, 128), (255, 128, 0), (0, 255, 0),
    ),
    "vaihingen": ((255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0)),
}
RGB_IGNORE = {
    "gid15": ((0, 0, 0), (0, 120, 64), (0, 140, 63), (0, 170, 64)),
    "hiertoulouse": (),
    "vaihingen": ((255, 0, 0),),
}


def label_palette(dataset):
    palette = {color: class_id for class_id, color in enumerate(RGB_CLASSES[dataset])}
    palette.update(dict.fromkeys(RGB_IGNORE[dataset], 255))
    return palette


def dataset_arguments(parser):
    parser.add_argument("--data", required=True, type=Path, help="Root with pre-split patches")
    parser.add_argument("--dataset", required=True, choices=CLASS_NAMES)


def model_arguments(parser):
    dataset_arguments(parser)
    parser.add_argument("--size", type=patch_size, default=256)
    parser.add_argument("--normalization", choices=["divide255", "minmax", "none"], default="divide255")


def patch_size(value):
    value = int(value)
    if value < 64 or value % 64:
        raise argparse.ArgumentTypeError("Use a positive multiple of 64 for the windowed encoder.")
    return value


def read_pair(pair, args, *, palette=None):
    image_path, mask_path = pair
    x = load_image(image_path, 3, args.normalization)
    y = load_label(mask_path, palette=palette) if mask_path is not None else None
    if x.shape[:2] != (args.size, args.size) or (y is not None and y.shape != x.shape[:2]):
        raise ValueError(f"Expected aligned {args.size} x {args.size} patches: {image_path}, {mask_path}")
    if y is not None:
        y[(y < 0) | (y >= len(CLASS_NAMES[args.dataset]))] = 255
    return x, y


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

