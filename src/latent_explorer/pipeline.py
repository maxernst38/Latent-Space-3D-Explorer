"""Dataset in, bundle out — the whole offline half in one call."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import DEFAULT_MODEL, DEFAULT_PRETRAINED, datasets
from .bundle import write_bundle
from .bundle.thumbnails import DEFAULT_TILE
from .embed.clip import encode_images, load_encoder
from .embed.reduce import project


def drop_unreadable(dataset: datasets.Dataset) -> list[str]:
    """Remove images Pillow can't open, returning what was dropped.

    Real datasets carry the occasional truncated or mislabelled file — Oxford
    Pets ships a handful. Discovering that a third of the way through an encode
    and losing the run is a bad trade for a check this cheap: Image.open only
    reads the header, so this is one stat per file, not a full decode.
    """
    keep, dropped = [], []
    for item in dataset.items:
        try:
            with Image.open(item.path) as image:
                image.verify()
            keep.append(item)
        except Exception:
            dropped.append(item.name)
    dataset.items = keep
    return dropped


def build(
    dataset_root: Path,
    out_dir: Path,
    fmt: str | None = None,
    split: str | None = None,
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    device: str | None = None,
    batch_size: int = 32,
    tile: int = DEFAULT_TILE,
    limit: int | None = None,
    thumbnails: bool = True,
    features_cache: Path | None = None,
    bundle_id: str | None = None,
    title: str | None = None,
    show_progress: bool = True,
) -> dict:
    dataset_root = Path(dataset_root)
    fmt = fmt or datasets.detect_format(dataset_root)
    dataset = datasets.load(dataset_root, fmt=fmt, split=split)

    if not len(dataset):
        raise ValueError(f"No images found in {dataset_root} (format: {fmt}).")

    dropped = drop_unreadable(dataset)
    if dropped:
        preview = ", ".join(dropped[:3])
        suffix = f" (+{len(dropped) - 3} more)" if len(dropped) > 3 else ""
        print(f"Skipped {len(dropped)} unreadable image(s): {preview}{suffix}")

    if not len(dataset):
        raise ValueError(f"Every image in {dataset_root} failed to open.")

    if limit and limit < len(dataset):
        # Stride rather than truncate, so a subset still spans every class in a
        # dataset that stores its classes contiguously — which most of them do.
        step = len(dataset) / limit
        keep = [dataset.items[int(i * step)] for i in range(limit)]
        dataset.items = keep

    # Re-encoding is the expensive step by a wide margin, so a cached feature
    # matrix lets you re-tune the projection or thumbnails for free.
    features = None
    if features_cache and Path(features_cache).is_file():
        features = np.load(features_cache)
        if features.shape[0] != len(dataset):
            features = None  # Cache is stale; the dataset changed underneath it.

    model_label = f"{model_name}/{pretrained}"
    if features is None:
        encoder = load_encoder(model_name, pretrained, device)
        model_label = encoder.label
        features = encode_images(
            dataset.paths, encoder, batch_size=batch_size, show_progress=show_progress
        )
        if features_cache:
            Path(features_cache).parent.mkdir(parents=True, exist_ok=True)
            np.save(features_cache, features)

    projection = project(features)

    return write_bundle(
        dataset=dataset,
        projection=projection,
        out_dir=out_dir,
        bundle_id=bundle_id,
        title=title,
        model_label=model_label,
        source_format=fmt,
        split=split,
        tile=tile,
        thumbnails=thumbnails,
        show_progress=show_progress,
    )
