"""Bundle round-trip: what the writer emits is what the viewer expects to read."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from latent_explorer import BUNDLE_FORMAT_VERSION, datasets
from latent_explorer.bundle import build_atlas, read_manifest, read_points, write_bundle
from latent_explorer.embed.reduce import Projection


@pytest.fixture
def dataset(tmp_path):
    """A tiny imagefolder dataset on disk, two classes of solid-colour images."""
    for class_name, colour in (("red", (200, 30, 30)), ("blue", (30, 30, 200))):
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(6):
            Image.new("RGB", (40, 30), colour).save(class_dir / f"{i}.jpg")
    return datasets.load(tmp_path, fmt="imagefolder")


@pytest.fixture
def projection(dataset):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(len(dataset), 3)).astype(np.float32)
    return Projection(points=points, explained_variance=[0.5, 0.3, 0.1])


def test_imagefolder_load_finds_classes_and_prefixes_names(dataset):
    assert dataset.class_names == ["blue", "red"]
    assert len(dataset) == 12
    assert all("/" in item.name for item in dataset.items)


def test_detect_format_recognises_an_imagefolder(dataset, tmp_path):
    assert datasets.detect_format(tmp_path) == "imagefolder"


def test_detect_format_rejects_a_directory_with_no_images(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(ValueError, match="Could not recognise"):
        datasets.detect_format(empty)


def test_bundle_round_trips(dataset, projection, tmp_path):
    out = tmp_path / "out"
    manifest = write_bundle(
        dataset, projection, out, model_label="test/none", show_progress=False
    )

    assert manifest["version"] == BUNDLE_FORMAT_VERSION
    assert manifest["count"] == 12
    assert {c["name"] for c in manifest["classes"]} == {"red", "blue"}

    # Positions survive the float32 binary trip at the format's stated precision.
    recovered = read_points(out)
    assert recovered.shape == (12, 3)
    assert np.allclose(recovered, projection.points, atol=1e-6)

    assert read_manifest(out)["id"] == "out"


def test_items_file_is_columnar_and_aligned(dataset, projection, tmp_path):
    out = tmp_path / "out"
    write_bundle(dataset, projection, out, show_progress=False)

    items = json.loads((out / "items.json").read_text())

    # Every column must be the same length as the point cloud, or the viewer
    # silently pairs a point with another image's label.
    assert len(items["name"]) == len(items["classes"]) == len(items["distance"]) == 12


def test_thumbnails_are_written_and_described(dataset, projection, tmp_path):
    out = tmp_path / "out"
    manifest = write_bundle(dataset, projection, out, tile=32, show_progress=False)

    atlas = manifest["thumbnails"]
    assert atlas["tile"] == 32
    assert atlas["count"] == 12
    for page in atlas["pages"]:
        assert (out / page).is_file()


def test_thumbnails_can_be_skipped(dataset, projection, tmp_path):
    out = tmp_path / "out"
    manifest = write_bundle(
        dataset, projection, out, thumbnails=False, show_progress=False
    )

    assert manifest["thumbnails"] is None
    assert not list(out.glob("thumbs-*.jpg"))


def test_atlas_tiles_are_laid_out_in_point_order(tmp_path):
    # Tile position is derived arithmetically by the viewer, so order is load-bearing.
    paths = []
    for i, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (10, 10), colour).save(path)
        paths.append(path)

    layout = build_atlas(paths, tmp_path / "atlas", tile=8, show_progress=False)
    sheet = Image.open(tmp_path / "atlas" / layout.pages[0])

    for index, expected in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255)]):
        col, row = index % layout.cols, index // layout.cols
        pixel = sheet.getpixel((col * 8 + 4, row * 8 + 4))
        assert all(abs(a - b) < 40 for a, b in zip(pixel, expected))


def test_unreadable_image_does_not_abort_the_atlas(tmp_path):
    good = tmp_path / "good.png"
    Image.new("RGB", (10, 10), (255, 0, 0)).save(good)
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")

    layout = build_atlas([good, broken], tmp_path / "atlas", tile=8, show_progress=False)

    assert layout.count == 2
    assert (tmp_path / "atlas" / layout.pages[0]).is_file()
