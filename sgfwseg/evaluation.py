"""Aggregate a confusion matrix over all labelled pixels in a split."""
import numpy as np
from PIL import Image

from .metrics import confusion, scores
from .protocol import CLASS_NAMES, read_pair


def evaluate(model, pairs, args, prediction_dir=None, *, batch_size=1, palette=None, progress=False):
    names = CLASS_NAMES[args.dataset]
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
    missing_labels = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        images, labels = zip(*(read_pair(pair, args, palette=palette) for pair in batch))
        predictions = model(np.stack(images), training=False).numpy().argmax(axis=-1)
        for pair, label, prediction in zip(batch, labels, predictions, strict=True):
            if label is None:
                missing_labels.append(pair[0].name)
            else:
                matrix += confusion(label, prediction, len(names))
            if prediction_dir is not None:
                Image.fromarray(prediction.astype(np.uint8)).save(prediction_dir / (pair[0].stem + ".png"))
        done = start + len(batch)
        if progress and (done == len(pairs) or done // 100 > start // 100):
            print(f"Predicted {done}/{len(pairs)} patches", flush=True)
    result = scores(matrix)
    if result["mIoU"] is None:
        raise ValueError("No evaluated class IDs found in the reference masks.")
    return dict(result, class_names=names, patches=len(pairs),
                evaluated_patches=len(pairs) - len(missing_labels), missing_labels=missing_labels)
