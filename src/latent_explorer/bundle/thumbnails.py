"""Packs every dataset image into tiled atlases the viewer uploads as textures.

One request per few thousand thumbnails instead of one per image. Without this
the hover preview would need N round trips through the single-threaded server
`view` runs, and the first hover on a large dataset would stall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

DEFAULT_TILE = 64
# Comfortably inside the WebGL texture limit on modest integrated graphics;
# anything larger risks a silent upload failure.
MAX_ATLAS_DIM = 4096


@dataclass
class AtlasLayout:
    tile: int
    cols: int
    rows: int
    pages: list[str]
    count: int

    @property
    def per_page(self) -> int:
        return self.cols * self.rows

    def to_json(self) -> dict:
        return {
            "tile": self.tile,
            "cols": self.cols,
            "rows": self.rows,
            "pages": self.pages,
            "count": self.count,
        }


def _square_thumbnail(path: Path, tile: int) -> Image.Image:
    """Centre-crop to a square, then downsize. Cropping rather than squashing
    keeps aspect ratios honest — a stretched preview misleads about the image."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        square = image.crop((left, top, left + side, top + side))
        return square.resize((tile, tile), Image.LANCZOS)


def build_atlas(
    paths: list[Path],
    out_dir: Path,
    tile: int = DEFAULT_TILE,
    quality: int = 82,
    show_progress: bool = True,
) -> AtlasLayout:
    """Write thumbs-N.jpg pages into `out_dir`, in the order `paths` is given.

    Tile order is point order, so the viewer maps point index to tile with
    arithmetic alone and needs no lookup table.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_row = MAX_ATLAS_DIM // tile
    cols = min(per_row, max(1, math.ceil(math.sqrt(len(paths)))))
    rows = min(per_row, max(1, math.ceil(len(paths) / cols)))
    per_page = cols * rows

    pages: list[str] = []
    page_count = max(1, math.ceil(len(paths) / per_page))

    iterator = enumerate(paths)
    if show_progress:
        iterator = tqdm(iterator, total=len(paths), desc="Thumbnails", unit="img")

    canvas: Image.Image | None = None
    current_page = -1

    for index, path in iterator:
        page = index // per_page
        if page != current_page:
            if canvas is not None:
                pages.append(_save_page(canvas, out_dir, current_page, quality))
            # The last page is usually partly empty; size it to what it holds
            # rather than padding out to a full sheet of black.
            remaining = len(paths) - page * per_page
            page_rows = min(rows, math.ceil(remaining / cols))
            canvas = Image.new("RGB", (cols * tile, page_rows * tile), (18, 18, 20))
            current_page = page

        slot = index % per_page
        try:
            thumb = _square_thumbnail(path, tile)
        except Exception:
            # An unreadable image leaves its tile as background rather than
            # aborting an export that is otherwise fine.
            continue
        canvas.paste(thumb, ((slot % cols) * tile, (slot // cols) * tile))

    if canvas is not None:
        pages.append(_save_page(canvas, out_dir, current_page, quality))

    assert len(pages) == page_count or not paths
    return AtlasLayout(tile=tile, cols=cols, rows=rows, pages=pages, count=len(paths))


def _save_page(canvas: Image.Image, out_dir: Path, page: int, quality: int) -> str:
    name = f"thumbs-{page}.jpg"
    canvas.save(out_dir / name, "JPEG", quality=quality, optimize=True)
    return name
