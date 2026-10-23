# The bundle format

A **bundle** is a directory of static files holding one dataset's finished
analysis. It is the contract between the two halves of this project: Python
writes it, JavaScript reads it, and nothing else crosses the line.

Everything expensive — CLIP inference, PCA, centroids, hulls, principal axes —
happens once, at build time. The viewer does no analysis; it draws what the
bundle already says. That is why `view` starts instantly however long the build
took, and why a bundle is worth keeping on its own: it is the result, a couple
of megabytes, re-openable without the dataset or the GPU that produced it.

Written by `latent_explorer.bundle.writer.write_bundle`, read by
`web/src/loadBundle.js`. The format is versioned by `BUNDLE_FORMAT_VERSION` in
`src/latent_explorer/__init__.py`; the viewer refuses a version it doesn't know
rather than misreading it silently.

```
bundles/oxford-pets/
├── manifest.json   everything derived
├── points.f32      raw float32 positions
├── items.json      per-image columns
└── thumbs-0.jpg    thumbnail atlas pages
```

## points.f32

Raw little-endian `float32`, `count × 3` values, x/y/z interleaved. Point *i*
occupies offsets `3i, 3i+1, 3i+2`.

Binary rather than JSON for two reasons: it is roughly five times smaller, and
it can be handed to a GPU buffer without a parse step. For 1480 points that is
18 kB against ~90 kB.

Positions are PCA output, recentred, then divided by the **98th-percentile
radius** — so the bulk of the cloud fills the unit sphere while genuine outliers
sit outside it. Scaling by the maximum instead would let a single extreme point
shrink everything else into a speck.

## items.json

Columnar, not an array of objects — three short arrays compress far better than
N little records, and the viewer indexes them directly.

```json
{
  "name":     ["Beagle/Beagle_1.jpg", ...],
  "classes":  [[4], [4], [12, 3], ...],
  "distance": [0.31, 0.88, null, ...]
}
```

Every column has exactly `count` entries, in the same order as `points.f32`.
That alignment is load-bearing: a mismatch pairs a point with another image's
label, and nothing would report an error.

- **`classes`** — class ids, as a list. Multi-label is normal for detection
  datasets, where one image contains several classes.
- **`distance`** — distance to the nearest centroid among the image's own
  classes, or `null` when it has no labels. This drives outlier mode.

## thumbs-N.jpg

Square thumbnails packed into a grid, in point order, so tile position is pure
arithmetic — there is no lookup table, and none is needed:

```
page = floor(i / (cols * rows))
slot = i % (cols * rows)
col  = slot % cols
row  = floor(slot / cols)
```

Pages exist because a 20 000-image dataset would need a sheet larger than the
WebGL texture limit; each page is capped at 4096 px. Images are centre-cropped
rather than squashed, so a preview never misrepresents its aspect ratio.

## manifest.json

```jsonc
{
  "version": 1,
  "id": "oxford-pets",
  "name": "Oxford-IIIT Pets · 37 breeds",
  "count": 1480,
  "createdAt": "2026-08-21T20:45:57+00:00",
  "source": { "format": "imagefolder", "split": null },
  "model": "ViT-B-32/laion2b_e16",

  "projection": {
    "kind": "pca",
    "explainedVariance": [0.11, 0.08, 0.06],
    "totalExplained": 0.252
  },

  // 95th percentile of distance. Above this, outlier mode lights a point up.
  // A percentile, not a constant: absolute scale still differs per dataset.
  "outlierThreshold": 0.83,

  "classes": [
    {
      "id": 4,
      "name": "Beagle",
      "count": 40,
      "centroid": [0.12, -0.44, 0.08],

      // First principal component of this class, in the projected 3D space —
      // so the axis points along spread the eye can actually verify.
      "axis":   [0.51, 0.20, 0.83],
      "extent": [-0.42, 0.39],
      "eigenImages": { "low": [112, 98, ...], "high": [141, 7, ...] },

      // Precomputed here because the class list is fixed: there is nothing for
      // the viewer to recompute, and it needs no geometry library.
      "hull": {
        "vertices": [x, y, z, x, y, z, ...],  // flat, corners only
        "faces":    [i, j, k, ...],           // flat, indexes `vertices`
        "volume": 0.0184
      }
    }
  ],

  "points":     { "file": "points.f32", "dtype": "float32", "stride": 3 },
  "items":      { "file": "items.json" },
  "thumbnails": { "tile": 64, "cols": 39, "rows": 38, "pages": [...], "count": 1480 }
}
```

Optional per class, and absent rather than null when they don't apply:

| Field | Omitted when |
|---|---|
| `centroid` | the class has no members |
| `axis`, `extent`, `eigenImages` | fewer than 6 members — fitting a direction to fewer is fitting noise |
| `hull` | fewer than 4 members, or they're coplanar (qhull can't build a 3D hull) |

`thumbnails` is `null` when built with `--no-thumbnails`.

## Changing the format

Bump `BUNDLE_FORMAT_VERSION`, update `SUPPORTED_VERSION` in `loadBundle.js`,
and rebuild every bundle. The viewer's version check turns what would be a
confusing misrender into a clear error message, so don't skip it.
