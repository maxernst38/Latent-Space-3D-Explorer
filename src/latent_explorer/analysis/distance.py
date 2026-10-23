"""How far each image sits from where its label says it belongs.

This is the number that turns the plot into a diagnostic. An image whose CLIP
embedding lands far from its own class centroid is either mislabelled, an
unusual example, or evidence that the class boundary is blurrier than the
label set pretends — and all three are worth looking at.
"""

from __future__ import annotations

import numpy as np

from ..datasets import Item


def distance_to_own_centroid(
    points: np.ndarray, items: list[Item], centroids: np.ndarray
) -> np.ndarray:
    """(N,) distance from each point to its own class centroid.

    Multi-label images are scored against the *nearest* of their labelled
    classes — flagging an image as an outlier because it sits far from one of
    three things it legitimately contains would be noise, not signal.

    Unlabelled images get NaN, which the bundle writer serialises as null.
    """
    distances = np.full(len(items), np.nan, dtype=np.float32)

    for index, item in enumerate(items):
        valid = [
            c for c in item.classes
            if 0 <= c < len(centroids) and not np.isnan(centroids[c]).any()
        ]
        if not valid:
            continue
        deltas = centroids[valid] - points[index]
        distances[index] = float(np.linalg.norm(deltas, axis=1).min())

    return distances


def outlier_threshold(distances: np.ndarray, percentile: float = 95.0) -> float:
    """Distance above which the viewer's outlier mode lights a point up.

    A percentile rather than a fixed number, because the absolute scale differs
    per dataset even after the projection is normalised.
    """
    finite = distances[np.isfinite(distances)]
    if not len(finite):
        return float("inf")
    return float(np.percentile(finite, percentile))
