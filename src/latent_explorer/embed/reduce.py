"""Projecting CLIP features down to the three dimensions the viewer draws."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class Projection:
    points: np.ndarray  # (N, 3) float32
    explained_variance: list[float]
    kind: str = "pca"

    @property
    def total_explained(self) -> float:
        return float(sum(self.explained_variance))


def project(features: np.ndarray, n_components: int = 3) -> Projection:
    """PCA down to `n_components`, recentred and scaled to a predictable box.

    The viewer's camera starts at a fixed distance, so raw PCA output — whose
    scale depends entirely on the dataset — would frame some datasets as a
    distant speck and others as a wall of points. Normalising here means the
    default camera is correct for every bundle.
    """
    if features.shape[0] <= n_components:
        raise ValueError(
            f"Need more than {n_components} images to project to "
            f"{n_components}D, got {features.shape[0]}."
        )

    pca = PCA(n_components=n_components)
    points = pca.fit_transform(features).astype(np.float32)

    points -= points.mean(axis=0)

    # Scale so that the *bulk* of the cloud fills the unit sphere, using the
    # 98th percentile radius rather than the maximum. Dividing by the farthest
    # point would let a handful of extreme outliers — exactly what this tool
    # exists to surface — shrink everything else into a speck at the centre.
    # The few points beyond radius 1 are meant to look extreme.
    radii = np.linalg.norm(points, axis=1)
    scale = float(np.percentile(radii, 98))
    if scale > 0:
        points /= scale

    return Projection(
        points=points,
        explained_variance=[float(v) for v in pca.explained_variance_ratio_],
    )
