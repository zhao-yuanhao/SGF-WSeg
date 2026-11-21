"""Evaluate one SGF-WSeg checkpoint and export class-ID predictions."""
import argparse
from pathlib import Path
import re

from sgfwseg.evaluation import evaluate
from sgfwseg.io import paired_files, save_color_predictions
from sgfwseg.protocol import CLASS_NAMES, RGB_CLASSES, label_palette, model_arguments, write_json


def vaihingen_label_key(stem):
    match = re.fullmatch(r"(.+)-(\d+)-\d+-(\d+)-\d+", stem)
    return "_".join(match.groups()) if match else stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-missing-labels", action="store_true",
                        help="Predict all inputs and score only patches with reference labels")
    parser.add_argument("--colorize", action="store_true",
                        help="Also export RGB masks with the dataset's reference palette")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.weights.is_file():
        parser.error(f"Checkpoint does not exist: {args.weights}")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new or empty evaluation directory.")
    flat = (args.data / "images").is_dir()
    pairs = paired_files(
        args.data if flat else args.data / "test",
        require_labels=not args.allow_missing_labels,
        label_folder="label" if flat else "labels",
        image_key=vaihingen_label_key if flat and args.dataset == "vaihingen" else None,
    )
    if not any(label is not None for _, label in pairs):
        parser.error("No reference labels matched the images.")

    from sgfwseg.model import load_sgfwseg
    import tensorflow as tf

    model = load_sgfwseg(args.weights, scale=args.size, num_class=len(CLASS_NAMES[args.dataset]))

    @tf.function(reduce_retracing=True)
    def predict(images, training=False):
        return model(images, training=training)

    prediction_dir = args.out / "predictions"
    result = evaluate(predict, pairs, args, prediction_dir=prediction_dir,
                      batch_size=args.batch_size, palette=label_palette(args.dataset), progress=True)
    if flat or args.colorize:
        save_color_predictions(prediction_dir, args.out / "predictions_rgb", RGB_CLASSES[args.dataset])
    result.update(dataset=args.dataset, checkpoint=args.weights.name, normalization=args.normalization)
    write_json(args.out / "metrics.json", result)
    print(f"Test OA {result['OA']:.4f} | mIoU {result['mIoU']:.4f} | "
          f"{result['evaluated_patches']}/{len(pairs)} patches evaluated")


if __name__ == "__main__":
    main()
