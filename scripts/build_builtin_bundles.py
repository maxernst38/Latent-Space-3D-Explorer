#!/usr/bin/env python
"""Builds the bundles that ship with the repo.

Two datasets, chosen so the pair says something neither says alone:

  * Oxford-IIIT Pets, cats vs dogs — two classes that separate almost
    perfectly. This is what a well-posed problem looks like in the projection.
  * CIFAR-10 — ten classes at 32x32, several of them genuinely confusable
    (cat/dog, automobile/truck, airplane/bird). Those clusters overlap, and the
    overlap is the interesting part rather than a failure of the plot.

    python scripts/build_builtin_bundles.py

Downloads ~950 MB on first run and caches it under .cache/ (gitignored).
"""

from __future__ import annotations

import pickle
import shutil
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from latent_explorer.pipeline import build  # noqa: E402

CACHE = REPO / ".cache"

PETS_URL = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
PETS_TARBALL = CACHE / "images.tar.gz"
PETS_IMAGES = CACHE / "images"

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_TARBALL = CACHE / "cifar-10-python.tar.gz"
CIFAR_BATCHES = CACHE / "cifar-10-batches-py"

# Enough per class for the shape of a cluster to be real, few enough that the
# bundles stay a couple of megabytes each.
PETS_PER_BREED = 40
CIFAR_PER_CLASS = 150


def download(url: str, target: Path, note: str) -> None:
    if target.is_file():
        return
    CACHE.mkdir(exist_ok=True)
    print(f"Downloading {url} ({note})…")
    urllib.request.urlretrieve(url, target)


def extract(tarball: Path, marker: Path) -> None:
    if marker.exists():
        return
    print(f"Extracting {tarball.name}…")
    with tarfile.open(tarball) as tar:
        # filter="data" refuses absolute paths and traversal entries; the
        # default became an error in 3.14 and a warning before that.
        tar.extractall(CACHE, filter="data")


# ── Oxford-IIIT Pets ────────────────────────────────────────────────────────


def breed_of(path: Path) -> str:
    """'american_bulldog_100.jpg' → 'american_bulldog'."""
    return path.stem.rsplit("_", 1)[0]


def is_cat(breed: str) -> bool:
    """Oxford Pets capitalises its 12 cat breeds and lowercases the 25 dog ones."""
    return breed[0].isupper()


def sample_pets() -> dict[str, list[Path]]:
    by_breed: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(PETS_IMAGES.glob("*.jpg")):
        by_breed[breed_of(path)].append(path)
    # Evenly spaced rather than the first N, so the sample isn't biased toward
    # whatever ordering the photographers happened to upload in.
    return {
        breed: [paths[int(i * len(paths) / PETS_PER_BREED)] for i in range(PETS_PER_BREED)]
        for breed, paths in sorted(by_breed.items())
        if len(paths) >= PETS_PER_BREED
    }


def stage_pets(sample: dict[str, list[Path]], root: Path, label) -> Path:
    """Lay the sample out as an imagefolder of symlinks.

    Symlinks into the cache rather than copies, and reusing the real
    imagefolder loader rather than a bespoke one — so this script exercises the
    same code path a user's own dataset would.
    """
    if root.exists():
        shutil.rmtree(root)

    for breed, paths in sample.items():
        class_dir = root / label(breed)
        class_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            (class_dir / path.name).symlink_to(path.resolve())

    return root


# ── CIFAR-10 ────────────────────────────────────────────────────────────────


def stage_cifar(root: Path) -> Path:
    """Materialise an even sample of CIFAR-10 as an imagefolder of PNGs.

    Unlike Pets, CIFAR ships as pickled uint8 arrays rather than files on disk,
    so these have to be written out rather than symlinked. PNG and not JPEG: at
    32x32 a lossy re-encode is a meaningful fraction of the signal being
    measured, and the whole sample is only a few megabytes either way.
    """
    if root.exists():
        return root

    import numpy as np
    from PIL import Image

    with open(CIFAR_BATCHES / "batches.meta", "rb") as fh:
        names = [n.decode() for n in pickle.load(fh, encoding="bytes")[b"label_names"]]

    by_class: dict[int, list[np.ndarray]] = defaultdict(list)
    for index in range(1, 6):
        with open(CIFAR_BATCHES / f"data_batch_{index}", "rb") as fh:
            batch = pickle.load(fh, encoding="bytes")
        # 3072 bytes per row: 1024 R, then G, then B, each row-major.
        images = batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        for image, label in zip(images, batch[b"labels"]):
            by_class[label].append(image)

    for label, images in sorted(by_class.items()):
        class_dir = root / names[label].replace("_", " ").title()
        class_dir.mkdir(parents=True, exist_ok=True)
        # Evenly spaced across the pooled batches, for the same reason as Pets.
        for i in range(CIFAR_PER_CLASS):
            picked = images[int(i * len(images) / CIFAR_PER_CLASS)]
            Image.fromarray(picked).save(class_dir / f"{i:04d}.png")

    return root


def main() -> int:
    download(PETS_URL, PETS_TARBALL, "~800 MB")
    extract(PETS_TARBALL, PETS_IMAGES)
    download(CIFAR_URL, CIFAR_TARBALL, "~163 MB")
    extract(CIFAR_TARBALL, CIFAR_BATCHES)

    pets = sample_pets()
    print(f"Sampled {sum(len(v) for v in pets.values())} pet images\n")

    print("── Bundle 1/2: cats vs dogs ──")
    build(
        dataset_root=stage_pets(
            pets, CACHE / "staging-species", lambda b: "Cat" if is_cat(b) else "Dog"
        ),
        out_dir=REPO / "bundles" / "oxford-pets-species",
        fmt="imagefolder",
        bundle_id="oxford-pets-species",
        title="Oxford-IIIT Pets · cats vs dogs",
        tile=48,
    )

    print("\n── Bundle 2/2: CIFAR-10 ──")
    build(
        dataset_root=stage_cifar(CACHE / "staging-cifar10"),
        out_dir=REPO / "bundles" / "cifar10",
        fmt="imagefolder",
        bundle_id="cifar10",
        title="CIFAR-10 · 10 classes",
        tile=48,
    )

    print("\nDone. Preview both:")
    print("  latent-explorer view bundles/oxford-pets-species bundles/cifar10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
