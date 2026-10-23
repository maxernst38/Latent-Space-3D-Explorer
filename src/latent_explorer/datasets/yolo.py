"""YOLO object-detection layout.

    root/
      dataset.yaml          names: [cat, dog, ...]
      images/<split>/*.jpg
      labels/<split>/*.txt  one "<class_id> cx cy w h" per line

`<split>/` is optional — some exports drop images straight into images/.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .base import Dataset, Item, find_images


def looks_like_yolo(root: Path) -> bool:
    return (root / "images").is_dir() and (
        (root / "labels").is_dir() or (root / "dataset.yaml").is_file()
    )


def available_splits(root: Path) -> list[str]:
    images_dir = root / "images"
    if not images_dir.is_dir():
        return []
    return sorted(d.name for d in images_dir.iterdir() if d.is_dir())


def read_class_names(root: Path) -> list[str]:
    """Class names from dataset.yaml, accepting both shapes Ultralytics emits."""
    for candidate in ("dataset.yaml", "data.yaml"):
        path = root / candidate
        if not path.is_file():
            continue
        with open(path) as f:
            names = (yaml.safe_load(f) or {}).get("names", [])
        # Newer exports use {0: 'cat', 1: 'dog'}; older ones a plain list.
        if isinstance(names, dict):
            return [names[k] for k in sorted(names, key=int)]
        if isinstance(names, list):
            return list(names)
    return []


def _read_label_file(path: Path, class_count: int) -> tuple[int, ...]:
    if not path.is_file():
        return ()
    found: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                class_id = int(line.split()[0])
            except ValueError:
                # A malformed row shouldn't sink the whole dataset.
                continue
            if 0 <= class_id < class_count:
                found.add(class_id)
    return tuple(sorted(found))


def load(root: Path, split: str | None = None) -> Dataset:
    root = Path(root)
    class_names = read_class_names(root)

    images_dir = root / "images"
    labels_dir = root / "labels"
    if split:
        images_dir = images_dir / split
        labels_dir = labels_dir / split

    items = []
    for image_path in find_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        items.append(
            Item(
                path=image_path,
                name=image_path.name,
                classes=_read_label_file(label_path, len(class_names)),
            )
        )

    name = root.name if not split else f"{root.name}/{split}"
    return Dataset(name=name, class_names=class_names, items=items)
