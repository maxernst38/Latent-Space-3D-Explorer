"""One directory per class — the layout most classification datasets ship in.

    root/
      Abyssinian/*.jpg
      Bengal/*.jpg

Also accepts a flat directory of loose images, which produces a valid unlabelled
dataset: the geometry is still worth looking at even with no classes to colour by.
"""

from __future__ import annotations

from pathlib import Path

from .base import Dataset, Item, find_images


def looks_like_imagefolder(root: Path) -> bool:
    if not root.is_dir():
        return False
    if find_images(root):
        return True
    return any(find_images(d) for d in root.iterdir() if d.is_dir())


def load(root: Path, split: str | None = None) -> Dataset:
    root = Path(root)
    if split:
        root = root / split

    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and find_images(d))
    class_names = [d.name for d in class_dirs]

    items = []
    for class_id, class_dir in enumerate(class_dirs):
        for image_path in find_images(class_dir):
            items.append(
                Item(
                    path=image_path,
                    # Prefixed so names stay unique across classes — plenty of
                    # datasets restart numbering inside every folder.
                    name=f"{class_dir.name}/{image_path.name}",
                    classes=(class_id,),
                )
            )

    # Loose images at the top level, with no class directories at all.
    if not items:
        items = [Item(path=p, name=p.name) for p in find_images(root)]

    return Dataset(name=root.name, class_names=class_names, items=items)
