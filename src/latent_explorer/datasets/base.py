"""What every dataset format is flattened into before anything else touches it.

The rest of the pipeline only ever sees `Dataset`, so adding a new on-disk
layout means writing one loader function here-adjacent and registering it in
`__init__.py` — nothing in embed/, analysis/ or bundle/ changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Everything Pillow opens reliably without extra plugins. The original tool only
# ever globbed "*.jpg", which silently ignored perfectly good PNG datasets.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class Item:
    """One image, plus whichever classes its labels claim it contains."""

    path: Path
    name: str
    # A tuple, not an int: YOLO detection images routinely contain several
    # classes at once, and collapsing that to one label would be a lie.
    classes: tuple[int, ...] = ()


@dataclass
class Dataset:
    name: str
    class_names: list[str]
    items: list[Item] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def paths(self) -> list[Path]:
        return [item.path for item in self.items]

    def class_counts(self) -> list[int]:
        counts = [0] * len(self.class_names)
        for item in self.items:
            for class_id in item.classes:
                if 0 <= class_id < len(counts):
                    counts[class_id] += 1
        return counts


def find_images(directory: Path) -> list[Path]:
    """Every image directly inside `directory`, in a stable order.

    Sorted so that point index N means the same image across re-runs — bundles
    reference items by index, so an unstable order would silently corrupt them.
    """
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
