import hashlib
import math

import numpy as np
from scipy import ndimage


def stable_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:8], 'little')


def point_mask(label, classes, budget, seed, name, strategy='random', denominator='all',
               edge_distance=2, center_distance=5, balanced=True):
    """Exactly rounded per-tile budget; class quotas never silently add extra points.

    A deterministic class permutation and candidate permutation make budgets nested.
    Equal class quotas use round-robin selection; absent/short classes redistribute.
    All strategies share class quotas. Empty center/edge candidates fall back to that
    class's valid region, explicitly reported in the metadata.
    """
    valid = (label >= 0) & (label < classes)
    n = int(valid.sum() if denominator == 'valid' else label.size)
    count = min(int(math.floor(n * budget / 100 + .5)), int(valid.sum()))
    out = np.full(label.shape, 255, np.uint8)
    rng = np.random.default_rng(stable_seed(seed, name))
    if strategy not in ('random', 'center', 'edge'):
        raise ValueError(strategy)
    lists, fallback = {}, []
    for c in rng.permutation(classes):
        region = label == c
        flat = np.flatnonzero(region)
        if not flat.size:
            continue
        # Padding makes the image edge a boundary; ignore pixels are outside region.
        dist = ndimage.distance_transform_edt(np.pad(region, 1))[1:-1, 1:-1]
        candidate = flat if strategy == 'random' else np.flatnonzero(
            region & ((dist <= edge_distance) if strategy == 'edge' else (dist >= center_distance)))
        if not candidate.size:
            candidate = flat
            fallback.append(int(c))
        ordered = rng.permutation(candidate)
        remaining = rng.permutation(np.setdiff1d(flat, candidate, assume_unique=True))
        lists[int(c)] = np.concatenate([ordered, remaining])
    selected = []
    if balanced:
        level = 0
        while len(selected) < count:
            for c, values in lists.items():
                if level < len(values) and len(selected) < count:
                    selected.append(int(values[level]))
            level += 1
    else:
        if strategy != 'random':
            raise ValueError('Unbalanced sampling is currently defined only for random')
        selected = rng.permutation(np.flatnonzero(valid))[:count].tolist()
    out.flat[selected] = label.flat[selected]
    # Records candidate exhaustion, including partial fallback beyond preferred set.
    outside = 0
    if strategy != 'random':
        for c in lists:
            d = ndimage.distance_transform_edt(np.pad(label == c, 1))[1:-1, 1:-1]
            preferred = d <= edge_distance if strategy == 'edge' else d >= center_distance
            outside += int(np.sum((out == c) & ~preferred))
    return out, dict(requested=count, actual=len(selected), denominator=n,
                     per_class=[int((out == c).sum()) for c in range(classes)],
                     empty_candidate_classes=fallback, fallback_points=outside,
                     strategy=strategy, edge_distance=edge_distance, center_distance=center_distance)
