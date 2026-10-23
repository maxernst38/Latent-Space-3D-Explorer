"""Per-class centroids — the anchor every other analysis here hangs off."""

from __future__ import annotations

import numpy as np

from ..datasets import Item


def group_by_class(items: list[Item], n_classes: int) -> list[np.ndarray]:
    """Point indices belonging to each class.

    A multi-label image appears in every class it contains, which is the whole
    point: an image holding both a cat and a dog genuinely sits inside both
    clusters, and its distance to each is worth seeing.
    """
    buckets: list[list[int]] = [[] for _ in range(n_classes)]
    for index, item in enumerate(items):
        for class_id in item.classes:
            if 0 <= class_id < n_classes:
                buckets[class_id].append(index)
    return [np.asarray(b, dtype=np.int32) for b in buckets]


def compute_centroids(points: np.ndarray, groups: list[np.ndarray]) -> np.ndarray:
    """(n_classes, 3) mean position per class. Empty classes get NaN."""
    centroids = np.full((len(groups), points.shape[1]), np.nan, dtype=np.float32)
    for class_id, indices in enumerate(groups):
        if len(indices):
            centroids[class_id] = points[indices].mean(axis=0)
    return centroids
