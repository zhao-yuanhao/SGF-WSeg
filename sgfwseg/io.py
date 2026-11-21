from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

IMAGE_SUFFIXES = {'.png', '.tif', '.tiff', '.jpg', '.jpeg', '.npy'}

def read_array(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
    elif suffix in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    else:
        array = np.asarray(Image.open(path))
    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim == 3 and array.shape[0] <= 16 and array.shape[-1] > 16:
        array = np.moveaxis(array, 0, -1)
    return np.asarray(array)


def normalize_image(array, mode="divide255"):
    array = array.astype(np.float32, copy=False)
    if mode == "none":
        return array
    if mode == "divide255":
        return array / 255.0
    if mode == "minmax":
        flat = array.reshape(-1, array.shape[-1])
        lo, hi = flat.min(0), flat.max(0)
        return (array - lo) / np.maximum(hi - lo, 1e-6)
    raise ValueError(f"Unknown normalization mode: {mode}")


def index_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return {}
    result = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise ValueError(f"Duplicate image stem {path.stem!r} in {folder}")
            result[path.stem] = path
    return result


def paired_files(root, require_labels=True, label_folder="labels", limit=None, image_key=None):
    root = Path(root)
    images, labels = index_images(root / "images"), index_images(root / label_folder)
    keys = {name: image_key(name) if image_key else name for name in images}
    if len(set(keys.values())) != len(keys):
        raise ValueError(f"Multiple images map to the same label in {root}")
    names = set(images)
    if require_labels:
        missing = {name for name in names if keys[name] not in labels}
        if missing:
            raise FileNotFoundError(f"{len(missing)} images lack labels in {root / label_folder}: {sorted(missing)[:3]}")
    names = sorted(names)
    if not names:
        raise FileNotFoundError(f"No matching image/label files found below {root}")
    if limit is not None:
        if limit < 1:
            raise ValueError("Sample limit must be positive")
        names = names[:limit]
    return [(images[name], labels.get(keys[name])) for name in names]


def load_image(path, channels=3, normalization="divide255"):
    image = read_array(path)
    if image.shape[-1] < channels:
        raise ValueError(f"{path} has {image.shape[-1]} channels; expected {channels}")
    return normalize_image(image[..., :channels], normalization)


def load_label(path, palette=None):
    array = read_array(path)
    if array.shape[-1] == 3 and palette is not None:
        ids = np.full(array.shape[:2], -1, dtype=np.int64)
        for color, class_id in palette.items():
            ids[np.all(array == color, axis=-1)] = class_id
        if np.any(ids == -1):
            colors = np.unique(array[ids == -1], axis=0).tolist()
            raise ValueError(f"Unknown RGB label colors in {path}: {colors[:5]}")
        return ids
    if array.shape[-1] != 1:
        raise ValueError(f"Expected single-channel class IDs, not an RGB mask: {path}")
    return array[..., 0].astype(np.int64)


def save_color_predictions(predictions, output, colors):
    """Render saved class-ID masks as RGB using colors in class-ID order."""
    palette = np.asarray(colors, dtype=np.uint8)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, path in index_images(predictions).items():
        ids = load_label(path)
        if np.any((ids < 0) | (ids >= len(palette))):
            raise ValueError(f"Prediction contains out-of-range class IDs: {path}")
        Image.fromarray(palette[ids]).save(output / (name + ".png"))
