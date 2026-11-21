"""Train SGF-WSeg using point-labelled training patches and dense validation masks."""
import argparse
from pathlib import Path

import numpy as np

from sgfwseg.evaluation import evaluate
from sgfwseg.io import paired_files
from sgfwseg.protocol import CLASS_NAMES, model_arguments, read_pair, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument("--points", default="points", help="Folder inside train, or an absolute mask directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--out", type=Path, required=True, help="New run directory")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Choose a new or empty run directory.")
    train = paired_files(args.data / "train", label_folder=args.points)
    val = paired_files(args.data / "val")
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    from sgfwseg.model import SGFWSeg
    model = SGFWSeg(scale=args.size, num_class=len(CLASS_NAMES[args.dataset]))
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    best, stale, history = -1., 0, []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        points, loss_sum = 0, 0.
        for start in range(0, len(order), args.batch_size):
            batch = [read_pair(train[i], args) for i in order[start:start + args.batch_size]]
            x, y = (np.stack(parts) for parts in zip(*batch))
            valid = y != 255
            count = int(valid.sum())
            if not count:
                continue
            with tf.GradientTape() as tape:
                probabilities = tf.boolean_mask(model(x, training=True), valid)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y[valid], probabilities))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_sum += count * float(loss)
            points += count
        if not points:
            raise ValueError("No training points found. Check mask IDs, point budget and patch dimensions.")
        metrics = evaluate(model, val, args)
        current = metrics["mIoU"]
        if current > best:
            best, stale = current, 0
            model.save_weights(str(args.out / "best.weights.h5"))
        else:
            stale += 1
        model.save_weights(str(args.out / "last.weights.h5"))
        lr = float(optimizer.learning_rate.numpy())
        history.append(dict(epoch=epoch, train_point_CE=loss_sum / points, training_points=points,
                            val_OA=metrics["OA"], val_mIoU=current, learning_rate=lr))
        write_json(args.out / "history.json", history)
        print(f"Epoch {epoch}/{args.epochs} | point CE {loss_sum / points:.4f} | val mIoU {current:.4f}", flush=True)
        if stale and stale % 5 == 0:
            optimizer.learning_rate.assign(max(lr * 0.5, args.lr * 0.001))


if __name__ == "__main__":
    main()
