"""Everything derived from the projected points: centroids, outliers, axes, hulls."""

from .centroids import compute_centroids, group_by_class
from .distance import distance_to_own_centroid, outlier_threshold
from .eigen import ClassAxis, compute_axes, principal_axis
from .hulls import Hull, compute_hull, compute_hulls

__all__ = [
    "ClassAxis",
    "Hull",
    "compute_axes",
    "compute_centroids",
    "compute_hull",
    "compute_hulls",
    "distance_to_own_centroid",
    "group_by_class",
    "outlier_threshold",
    "principal_axis",
]
