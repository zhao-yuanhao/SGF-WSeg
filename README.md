# SGF-WSeg

TensorFlow implementation of **SGF-WSeg: Filtering- and Wavelet-Guided Weakly Supervised Land-Cover Segmentation for High-Resolution Remote Sensing Images**. This repository contains the proposed network, the point-mask generator, validation-based training and checkpoint evaluation.

## Environment

The commands below use Python 3.12 on Linux (including WSL2). Run them from the directory containing this README. A CPU is sufficient for a small functional test.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an NVIDIA GPU on Linux, install the CUDA-enabled dependencies into the same environment and check device visibility:

```bash
python -m pip install 'tensorflow[and-cuda]==2.21.0'
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The model uses the legacy TensorFlow Keras API. The command-line entry points select it automatically. For notebook use, import `sgfwseg` before importing TensorFlow in a fresh kernel.

## Prepare the data

Place pre-split, aligned patches in the following layout. The example uses Vaihingen. Use a separate root for each dataset and keep each source scene in one split. The programs consume the split supplied by the user.

```text
data/vaihingen/
  train/
    images/tile_a.tif
    labels-all/tile_a.png      # dense reference, needed only to simulate points
    points/tile_a.png          # generated below, or supplied sparse annotation
  val/
    images/tile_b.tif
    labels/tile_b.png          # dense reference
  test/
    images/tile_c.tif
    labels/tile_c.png          # dense reference
weights/                      # optional downloaded or author-supplied checkpoints
```

Images and masks are matched by filename stem within each split. Use three-channel imagery in the intended band order. The loader takes the first three channels and divides by 255 by default. For other radiometric scales, convert the input or pass `--normalization minmax` (per patch and channel) or `none` consistently to training and testing. Images may be PNG, JPEG, TIFF or NumPy arrays. Use lossless single-channel integer class-ID masks. Install a matching TIFF codec if your source uses optional TIFF compression. Use 256 × 256 patches for the paper setting. Other square sizes must be multiples of 64. The pipeline does not rescale a misaligned mask.

| Dataset flag | Evaluated IDs | Classes |
| --- | --- | --- |
| `vaihingen` | 0–4 | Impervious surfaces, buildings, low vegetation, trees, cars |
| `gid15` | 0–14 | 15 land-cover classes |
| `hiertoulouse` | 0–9 | 10 land-cover classes |

The exact class order is in [classes.json](sgfwseg/classes.json). Remap raw product IDs or palette colours into this order before use. Class 0 is evaluated. Use 255 for unlabelled or ignored pixels. IDs outside the selected preset are excluded from sampling, training and evaluation.

## Generate point annotations

Skip this step if you already have sparse training masks. For simulated point supervision, start with `train/images` and `train/labels-all`. The output directory must be new.

```bash
python generate_points.py --data data/vaihingen --dataset vaihingen --budget 0.01 --seed 2024
```

`--budget` is a **percentage**. At 0.01%, a fully valid 256 × 256 patch receives seven points after rounding. The budget is capped at the available valid pixels. A seed and patch ID determine the locations. The sampler cycles through present classes, redistributing unused quotas. The default is `--strategy random`. With `center` or `edge`, preferred pixels have a within-class distance to the region boundary of ≥5 or ≤2 pixels, respectively (padded Euclidean distance transform). If the preferred candidates run out, the remaining points come from other pixels of that class. The PNG outputs and `manifest.json` in `train/points` record the masks and per-patch class counts. Use `--output data/vaihingen/train/points_42` to preserve an additional draw.

## Train

```bash
python train.py --data data/vaihingen --dataset vaihingen --points points --seed 2024 --epochs 30 --batch-size 8 --lr 0.001 --out runs/vaihingen_2024
```

For supplied annotations in, for example, `train/labels-p-0.01%`, pass `--points 'labels-p-0.01%'`. Training does not need dense training masks when sparse masks are supplied. Keep the generator seed and the training seed in the run record. For the batch-size-six configuration used in several supplemental experiments, pass `--batch-size 6`.

The training loss is mean cross-entropy over the labelled pixels in each minibatch. Batches containing no labelled pixels are skipped. Adam starts at the requested learning rate. After each five consecutive epochs without a new best validation mIoU, the learning rate is halved down to 0.001 times its initial value. There is no data augmentation in this configuration.

Each run writes `config.json`, `history.json`, `best.weights.h5` (highest dense-validation mIoU) and `last.weights.h5` (final epoch). Choose a new output directory for each run.

## Test

```bash
python test.py --data data/vaihingen --dataset vaihingen --weights runs/vaihingen_2024/best.weights.h5 --out runs/vaihingen_2024/test_best
```

Use `last.weights.h5` and a different output directory to evaluate the last epoch. Match the dataset, patch size and normalization used for training.

For the supplied HDF5 checkpoints, use the corresponding dataset flag.

| File under `weights/` | Dataset flag | Output classes |
| --- | --- | --- |
| `GID.hdf5` | `gid15` | 15 |
| `HierToulouse.hdf5` | `hiertoulouse` | 10 |
| `ISPRS Vaihingen.hdf5` | `vaihingen` | 5 |

The supplied test patches use `images/` and `label/` within each dataset directory. The evaluator maps RGB labels to the dataset's class IDs.

```bash
python test.py --data data/GID --dataset gid15 --weights weights/GID.hdf5 --out data/GID/sgf_wseg
python test.py --data data/HierToulouse --dataset hiertoulouse --weights weights/HierToulouse.hdf5 --out data/HierToulouse/sgf_wseg
python test.py --data data/Vaihingen --dataset vaihingen --weights "weights/ISPRS Vaihingen.hdf5" --out data/Vaihingen/sgf_wseg --allow-missing-labels
```

Choose a fresh output directory to repeat an evaluation. Add `--batch-size 8` on a GPU with sufficient memory. The Vaihingen command saves predictions for all images and computes metrics over the images with matching references. The metrics file lists images without a reference label.

To load a supplied checkpoint directly in Python, pass its path. The loader reads the number of output classes from the checkpoint.

```python
from sgfwseg.model import load_sgfwseg

model = load_sgfwseg("weights/ISPRS Vaihingen.hdf5")
```

`metrics.json` contains OA, mean IoU, per-class IoU in class-ID order, and the accumulated confusion matrix (rows are reference IDs, columns are predictions). Metrics are fractions in [0, 1]. An absent class with zero union has a null IoU and is excluded from the mean. `predictions/` contains one class-ID PNG per test patch with the original stem. For the supplied layout, `predictions_rgb/` also contains RGB masks in the dataset's reference colors. Use `--colorize` to export RGB masks with the split layout.

For another dataset, change both `--data` and `--dataset` in all three commands. The method starts from random initialization, so no downloaded weights are needed for training.

