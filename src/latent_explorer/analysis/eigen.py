"""Eigen-images: what a class's axis of maximum variance actually looks like.

Each class has a direction along which its members spread out most. Walking
that direction and pulling out the images at either end shows you, concretely,
what the model thinks the class varies *in* — pose, lighting, background,
subject age. It answers "what does this cluster's shape mean?" with pictures
instead of numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Below this, "direction of maximum variance" is fitting a line to noise.
MIN_MEMBERS = 6


@dataclass
class ClassAxis:
    axis: np.ndarray  # (3,) unit vector through the centroid
    extent: tuple[float, float]  # how far members project along it, min/max
    low: list[int]  # point indices at the negative end
    high: list[int]  # point indices at the positive end

    @property
    def is_empty(self) -> bool:
        return not self.low and not self.high


def _empty_axis() -> ClassAxis:
    return ClassAxis(
        axis=np.zeros(3, dtype=np.float32), extent=(0.0, 0.0), low=[], high=[]
    )


def principal_axis(points: np.ndarray, indices: np.ndarray, k: int = 5) -> ClassAxis:
    """First principal component of one class, plus its extreme members.

    Computed in the projected 3D space rather than in CLIP's native
    dimensionality, deliberately: the viewer draws this axis as a line through
    the cloud, so it has to be the direction the spread is actually visible
    along. An axis derived in 512-D and then projected would point somewhere
    the eye cannot verify.
    """
    if len(indices) < MIN_MEMBERS:
        return _empty_axis()

    members = points[indices]
    centred = members - members.mean(axis=0)

    # SVD rather than sklearn's PCA — one component from a handful of 3D points
    # doesn't need the estimator machinery, and this stays exact for tiny inputs.
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return _empty_axis()

    axis = vt[0].astype(np.float32)

    # Sign of a principal component is arbitrary; pin it so re-running the
    # pipeline doesn't flip which end is "low" and which is "high".
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis

    projections = centred @ axis
    order = np.argsort(projections)
    take = min(k, len(indices) // 2)

    return ClassAxis(
        axis=axis,
        extent=(float(projections.min()), float(projections.max())),
        low=[int(indices[i]) for i in order[:take]],
        high=[int(indices[i]) for i in order[-take:][::-1]],
    )


def compute_axes(
    points: np.ndarray, groups: list[np.ndarray], k: int = 5
) -> list[ClassAxis]:
    return [principal_axis(points, indices, k) for indices in groups]
