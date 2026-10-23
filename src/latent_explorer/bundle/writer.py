"""Writes the static bundle that the viewer reads.

This module is the contract between the two halves of the project. Everything
upstream of it needs Python and a GPU; everything downstream is a static file
served from any web host. The format is documented in docs/bundle-format.md and
versioned by BUNDLE_FORMAT_VERSION — change the shape, bump the number, or the
viewer will misread older bundles without saying so.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import BUNDLE_FORMAT_VERSION
from ..analysis import (
    compute_axes,
    compute_centroids,
    compute_hulls,
    distance_to_own_centroid,
    group_by_class,
    outlier_threshold,
)
from ..datasets import Dataset
from ..embed.reduce import Projection
from .thumbnails import DEFAULT_TILE, build_atlas

# Positions live in [-1, 1] and thumbnails are 64px, so four decimals is well
# below one pixel of error — and roughly halves the size of the JSON.
PRECISION = 4


def _round(values) -> list:
    return [round(float(v), PRECISION) for v in np.asarray(values).ravel()]


def _class_entry(
    class_id: int,
    name: str,
    indices: np.ndarray,
    centroid: np.ndarray,
    axis,
    hull,
) -> dict:
    entry: dict = {
        "id": class_id,
        "name": name,
        "count": int(len(indices)),
    }
    if not np.isnan(centroid).any():
        entry["centroid"] = _round(centroid)
    if not axis.is_empty:
        entry["axis"] = _round(axis.axis)
        entry["extent"] = _round(axis.extent)
        entry["eigenImages"] = {"low": axis.low, "high": axis.high}
    if not hull.is_empty:
        # Flattened rather than nested triples: same data, noticeably less JSON.
        entry["hull"] = {
            "vertices": _round(hull.vertices),
            "faces": [int(i) for i in hull.faces.ravel()],
            "volume": round(hull.volume, 6),
        }
    return entry


def write_bundle(
    dataset: Dataset,
    projection: Projection,
    out_dir: Path,
    bundle_id: str | None = None,
    title: str | None = None,
    model_label: str = "unknown",
    source_format: str = "unknown",
    split: str | None = None,
    tile: int = DEFAULT_TILE,
    eigen_k: int = 5,
    thumbnails: bool = True,
    show_progress: bool = True,
) -> dict:
    """Compute every derived quantity and write the bundle. Returns the manifest."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points = projection.points
    n_classes = len(dataset.class_names)

    groups = group_by_class(dataset.items, n_classes)
    centroids = compute_centroids(points, groups)
    distances = distance_to_own_centroid(points, dataset.items, centroids)
    axes = compute_axes(points, groups, k=eigen_k)
    hulls = compute_hulls(points, groups)

    # Positions go out as raw float32 rather than JSON — for 1500 points that is
    # 18 kB instead of ~90 kB, and the viewer hands the buffer straight to the GPU.
    points.astype(np.float32).tofile(out_dir / "points.f32")

    # Columnar: three short arrays compress far better than N little objects.
    items_payload = {
        "name": [item.name for item in dataset.items],
        "classes": [list(item.classes) for item in dataset.items],
        "distance": [
            None if not np.isfinite(d) else round(float(d), PRECISION)
            for d in distances
        ],
    }
    (out_dir / "items.json").write_text(json.dumps(items_payload, separators=(",", ":")))

    atlas = None
    if thumbnails:
        atlas = build_atlas(
            dataset.paths, out_dir, tile=tile, show_progress=show_progress
        )

    manifest = {
        "version": BUNDLE_FORMAT_VERSION,
        "id": bundle_id or out_dir.name,
        "name": title or dataset.name,
        "count": len(dataset.items),
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"format": source_format, "split": split},
        "model": model_label,
        "projection": {
            "kind": projection.kind,
            "explainedVariance": [round(v, 6) for v in projection.explained_variance],
            "totalExplained": round(projection.total_explained, 6),
        },
        "outlierThreshold": round(outlier_threshold(distances), PRECISION),
        "classes": [
            _class_entry(
                i, dataset.class_names[i], groups[i], centroids[i], axes[i], hulls[i]
            )
            for i in range(n_classes)
        ],
        "points": {"file": "points.f32", "dtype": "float32", "stride": 3},
        "items": {"file": "items.json"},
        "thumbnails": atlas.to_json() if atlas else None,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_manifest(bundle_dir: Path) -> dict:
    return json.loads((Path(bundle_dir) / "manifest.json").read_text())


def read_points(bundle_dir: Path) -> np.ndarray:
    """Round-trip helper — mostly for tests and for re-inspecting a built bundle."""
    manifest = read_manifest(bundle_dir)
    raw = np.fromfile(Path(bundle_dir) / manifest["points"]["file"], dtype=np.float32)
    return raw.reshape(-1, manifest["points"]["stride"])
