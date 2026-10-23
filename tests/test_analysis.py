"""Analysis maths, checked against geometry chosen so the answers are known."""

from __future__ import annotations

import numpy as np
import pytest

from latent_explorer.analysis import (
    compute_axes,
    compute_centroids,
    compute_hulls,
    distance_to_own_centroid,
    group_by_class,
    outlier_threshold,
)
from latent_explorer.datasets import Item


def make_items(class_lists):
    return [Item(path=f"{i}.jpg", name=f"{i}.jpg", classes=tuple(c))
            for i, c in enumerate(class_lists)]


def test_group_by_class_puts_multilabel_items_in_every_class():
    items = make_items([[0], [1], [0, 1], []])
    groups = group_by_class(items, n_classes=2)

    assert groups[0].tolist() == [0, 2]
    assert groups[1].tolist() == [1, 2]


def test_centroid_is_the_mean_of_its_members():
    points = np.array([[0, 0, 0], [2, 0, 0], [10, 10, 10]], dtype=np.float32)
    groups = group_by_class(make_items([[0], [0], [1]]), n_classes=2)

    centroids = compute_centroids(points, groups)

    assert centroids[0].tolist() == [1.0, 0.0, 0.0]
    assert centroids[1].tolist() == [10.0, 10.0, 10.0]


def test_empty_class_gets_a_nan_centroid_not_an_origin_one():
    # A zero centroid would be indistinguishable from a real class sitting at
    # the origin, and would drag every distance calculation toward it.
    points = np.array([[1, 1, 1]], dtype=np.float32)
    groups = group_by_class(make_items([[0]]), n_classes=2)

    centroids = compute_centroids(points, groups)

    assert np.isnan(centroids[1]).all()


def test_distance_uses_the_nearest_of_several_labelled_classes():
    points = np.array([[0, 0, 0], [10, 0, 0], [1, 0, 0]], dtype=np.float32)
    items = make_items([[0], [1], [0, 1]])
    groups = group_by_class(items, n_classes=2)
    centroids = compute_centroids(points, groups)

    distances = distance_to_own_centroid(points, items, centroids)

    # Point 2 belongs to both classes; it is far from class 1's centroid but
    # close to class 0's, so it should not read as an outlier.
    assert distances[2] < 1.0


def test_unlabelled_items_have_no_distance():
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    items = make_items([[0], []])
    groups = group_by_class(items, n_classes=1)
    centroids = compute_centroids(points, groups)

    distances = distance_to_own_centroid(points, items, centroids)

    assert np.isnan(distances[1])


def test_outlier_threshold_ignores_nan():
    distances = np.array([1.0, 2.0, np.nan, 3.0, 100.0], dtype=np.float32)
    assert np.isfinite(outlier_threshold(distances, percentile=50))


def test_principal_axis_follows_the_direction_of_spread():
    # Points spread along x, tight in y and z — the axis must be ±x.
    points = np.array([[float(i), 0.0, 0.0] for i in range(10)], dtype=np.float32)
    groups = group_by_class(make_items([[0]] * 10), n_classes=1)

    axis = compute_axes(points, groups, k=2)[0]

    assert abs(abs(axis.axis[0]) - 1.0) < 1e-5
    assert abs(axis.axis[1]) < 1e-5


def test_eigen_images_are_the_extremes_along_that_axis():
    points = np.array([[float(i), 0.0, 0.0] for i in range(10)], dtype=np.float32)
    groups = group_by_class(make_items([[0]] * 10), n_classes=1)

    axis = compute_axes(points, groups, k=2)[0]

    # Whichever way the sign lands, the two ends must be the two extremes.
    assert set(axis.low) | set(axis.high) == {0, 1, 8, 9}


def test_axis_is_empty_for_a_class_too_small_to_fit_one():
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    groups = group_by_class(make_items([[0], [0]]), n_classes=1)

    assert compute_axes(points, groups)[0].is_empty


def test_hull_encloses_a_cube_and_indexes_only_its_corners():
    corners = np.array(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float32
    )
    interior = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    points = np.vstack([corners, interior])
    groups = group_by_class(make_items([[0]] * 9), n_classes=1)

    hull = compute_hulls(points, groups)[0]

    assert hull.volume == pytest.approx(1.0)
    assert len(hull.vertices) == 8  # the interior point is dropped
    assert hull.faces.max() < len(hull.vertices)


def test_degenerate_class_yields_an_empty_hull_rather_than_raising():
    # All coplanar — qhull cannot build a 3D hull from this.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
    groups = group_by_class(make_items([[0]] * 4), n_classes=1)

    assert compute_hulls(points, groups)[0].is_empty
