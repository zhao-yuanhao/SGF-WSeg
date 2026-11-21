"""Simulate sparse training annotations from dense class-ID masks."""
import argparse
import tempfile
from pathlib import Path

from PIL import Image

from sgfwseg.io import load_label, paired_files
from sgfwseg.protocol import CLASS_NAMES, dataset_arguments, write_json
from sgfwseg.sampling import point_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_arguments(parser)
    parser.add_argument("--budget", type=float, default=0.01, help="Percent of all pixels per patch")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--strategy", choices=["random", "center", "edge"], default="random")
    parser.add_argument("--dense-folder", default="labels-all")
    parser.add_argument("--output", type=Path, help="New directory, default DATA/train/points")
    parser.add_argument("--edge-distance", type=float, default=2)
    parser.add_argument("--center-distance", type=float, default=5)
    args = parser.parse_args()
    if not 0 < args.budget <= 100 or not (0 <= args.edge_distance <= args.center_distance):
        parser.error("Use 0 < budget <= 100 and 0 <= edge-distance <= center-distance.")
    train = args.data / "train"
    target = args.output or train / "points"
    if target.exists():
        parser.error(f"Output exists: {target}. Choose a new output for each set of masks.")
    pairs = paired_files(train, label_folder=args.dense_folder)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
        stage = Path(temporary) / "masks"
        stage.mkdir()
        for image, label in pairs:
            mask, stats = point_mask(load_label(label), len(CLASS_NAMES[args.dataset]),
                                     args.budget, args.seed, f"train/{image.stem}",
                                     args.strategy, "all", args.edge_distance, args.center_distance)
            Image.fromarray(mask).save(stage / (image.stem + ".png"))
            records.append(dict(id=f"train/{image.stem}", **stats))
        write_json(stage / "manifest.json", dict(dataset=args.dataset, seed=args.seed,
                   budget_percent=args.budget, strategy=args.strategy, records=records))
        stage.rename(target)
    print(f"{len(records)} masks, {sum(row['actual'] for row in records)} points: {target}")


if __name__ == "__main__":
    main()
