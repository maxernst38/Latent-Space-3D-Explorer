"""Dataset loaders and format autodetection."""

from __future__ import annotations

from pathlib import Path

from . import imagefolder, yolo
from .base import IMAGE_EXTENSIONS, Dataset, Item, find_images

FORMATS = ("yolo", "imagefolder")

__all__ = [
    "Dataset",
    "Item",
    "IMAGE_EXTENSIONS",
    "FORMATS",
    "find_images",
    "detect_format",
    "available_splits",
    "load",
]


def detect_format(root: Path) -> str:
    """Guess the on-disk layout. YOLO is checked first — a YOLO root also has
    directories under it, so imagefolder would otherwise match it by accident."""
    root = Path(root)
    if yolo.looks_like_yolo(root):
        return "yolo"
    if imagefolder.looks_like_imagefolder(root):
        return "imagefolder"
    raise ValueError(
        f"Could not recognise a dataset in {root}. Expected either a YOLO layout "
        f"(images/ + labels/) or one directory per class."
    )


def available_splits(root: Path, fmt: str | None = None) -> list[str]:
    root = Path(root)
    fmt = fmt or detect_format(root)
    if fmt == "yolo":
        return yolo.available_splits(root)
    return []


def load(root: Path, fmt: str | None = None, split: str | None = None) -> Dataset:
    root = Path(root)
    fmt = fmt or detect_format(root)
    if fmt == "yolo":
        return yolo.load(root, split)
    if fmt == "imagefolder":
        return imagefolder.load(root, split)
    raise ValueError(f"Unknown dataset format: {fmt!r}. Expected one of {FORMATS}.")
