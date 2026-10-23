"""Convex hulls around each class, precomputed as drawable meshes.

Hulls are built here rather than in the browser because the class list is fixed
at export time — there is nothing for the viewer to recompute when someone
toggles a class on. That keeps scipy on this side of the line and spares the
web bundle a computational-geometry library entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

# A 3D hull needs a non-degenerate tetrahedron at minimum.
MIN_POINTS = 4


@dataclass
class Hull:
    vertices: np.ndarray  # (V, 3) float32 — hull corners only, not every member
    faces: np.ndarray  # (F, 3) int32 — triangles indexing into `vertices`
    volume: float

    @property
    def is_empty(self) -> bool:
        return len(self.faces) == 0


def _empty_hull() -> Hull:
    return Hull(
        vertices=np.zeros((0, 3), dtype=np.float32),
        faces=np.zeros((0, 3), dtype=np.int32),
        volume=0.0,
    )


def compute_hull(points: np.ndarray, indices: np.ndarray) -> Hull:
    """Convex hull of one class, reindexed down to just its corner vertices.

    Returns an empty hull rather than raising when the class is too small or
    its points are coplanar — a class that cannot be enclosed is a normal
    outcome here, not an error worth stopping an export for.
    """
    if len(indices) < MIN_POINTS:
        return _empty_hull()

    members = points[indices]
    try:
        hull = ConvexHull(members)
    except Exception:
        # QhullError for coplanar/degenerate inputs, which are common in
        # tight classes. Skip the hull, keep the class.
        return _empty_hull()

    # scipy indexes simplices into `members`; most of those points are interior
    # and never referenced. Remap onto the corners alone so the mesh shipped to
    # the browser carries a few dozen vertices instead of a few thousand.
    corners = hull.vertices
    remap = np.full(len(members), -1, dtype=np.int32)
    remap[corners] = np.arange(len(corners), dtype=np.int32)

    return Hull(
        vertices=members[corners].astype(np.float32),
        # Winding is not consistently outward, so the viewer renders these
        # double-sided; that also reads better for a translucent shell.
        faces=remap[hull.simplices].astype(np.int32),
        volume=float(hull.volume),
    )


def compute_hulls(points: np.ndarray, groups: list[np.ndarray]) -> list[Hull]:
    return [compute_hull(points, indices) for indices in groups]
