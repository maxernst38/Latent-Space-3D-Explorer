# Latent Space 3D Explorer

Encodes an image dataset with CLIP, projects the embeddings down to three
dimensions, and lets you fly through the result in a browser — where visual
similarity becomes physical distance.

It goes past plotting points. It computes per-class centroids, measures how far
each image sits from where its label says it belongs, and surfaces
**eigen-images**: the pictures at either end of a class's axis of maximum
variance. The point is to make dataset problems *visible* — mislabelled samples
and blurred class boundaries show up as spatial outliers rather than as a number
in an evaluation table.

It's a local tool. Two commands: one to analyse a dataset, one to look at it.

```bash
uv run latent-explorer build datasets/my-set -o bundles/my-set
uv run latent-explorer view bundles/my-set
```

---

## Build once, view many times

CLIP inference needs Python, a few gigabytes of PyTorch, and ideally a GPU.
Orbiting a point cloud needs a browser. So the two halves are separated by a
**bundle** — a directory of static files holding the finished analysis:

```
   your dataset ──▶ [ Python: CLIP → PCA → centroids,     ] ──▶ bundle/ ──▶ [ three.js viewer ]
                    [ distances, hulls, principal axes    ]                  `view` serves it
```

Everything expensive happens once, at `build` time. The viewer does no
analysis — it draws what the bundle already says. So `view` starts instantly
however long the build took, and a bundle is worth keeping on its own: it's the
result, a couple of megabytes, re-openable without the dataset or the GPU that
produced it.

`view` is a static file server on loopback. Nothing is uploaded, and nothing
runs that you didn't start.

The format is documented in [docs/bundle-format.md](docs/bundle-format.md).

---

## Install

PyTorch has no Python 3.14 wheels yet, so the project pins 3.12. With
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Max-Ernst/Latent-Space-3D-Explorer.git
cd Latent-Space-3D-Explorer
uv sync                     # installs Python 3.12 and everything else
```

A CUDA GPU is used when present, then Apple Silicon MPS, then CPU. CPU works
fine — 1500 images take a couple of minutes rather than five seconds.

## Look at something immediately

Two bundles are committed, so there's something to explore before you've built
anything of your own:

```bash
uv run latent-explorer view bundles/oxford-pets-species bundles/cifar10
```

Cats vs dogs separate almost perfectly — that's what a well-posed problem looks
like in the projection. CIFAR-10's ten classes don't, and the overlap
(cat/dog, automobile/truck, airplane/bird) is the interesting part rather than a
failure of the plot.

## Use it on your own dataset

```bash
uv run latent-explorer build datasets/my-set -o bundles/my-set
uv run latent-explorer view bundles/my-set
```

`build` autodetects either layout:

**One directory per class** — most classification datasets:

```
my-set/
  Abyssinian/*.jpg
  Bengal/*.jpg
```

**YOLO object detection** — `--split` picks one when present:

```
my-set/
  dataset.yaml          names: [cat, dog, ...]
  images/train/*.jpg
  labels/train/*.txt
```

A flat directory of loose images also works; you get the geometry with no
classes to colour by.

Useful flags:

| Flag | |
|---|---|
| `--limit N` | Sample N images, evenly spaced across the set |
| `--model` / `--pretrained` | Any [open_clip](https://github.com/mlfoundations/open_clip) pair. Default `ViT-B-32` / `laion2b_e16` |
| `--cache path.npy` | Reuse the CLIP features, to re-tune everything downstream for free |
| `--no-thumbnails` | Skip the atlas — much smaller, but hovering shows only a filename |
| `--device` | Force `cuda`, `mps` or `cpu` |

`view` takes several bundles at once and puts a dataset picker in the header,
which is how you compare two runs over the same data — a different model, say,
or a tighter `--limit`.

`latent-explorer inspect bundles/my-set` summarises one without opening it.

## What you can do in the viewer

- **Drag** to orbit, scroll to zoom. Faint axes cross at the origin — the
  centre of the cloud — so orbiting doesn't lose you.
- **Hover** any point for the actual image, its labels, and its distance from
  its class centroid.
- **Pick classes** to colour them, up to eight at once. Toggle centroids,
  convex **hulls**, and the **variance axis**.
- **Outliers** shades every image by how far it sits from its own centroid and
  enlarges the worst 5%. This is where mislabelled data tends to show up.
- With exactly one class selected, the **eigen-image strip** appears: the
  pictures at either end of that class's longest dimension.

---

## Repository layout

```
src/latent_explorer/      the offline half — needs Python
  datasets/               dataset layouts (yolo, imagefolder) behind one protocol
  embed/                  CLIP encoding; PCA projection
  analysis/               centroids, distances, eigen axes, convex hulls
  bundle/                 writes the static bundle; packs the thumbnail atlas
  cli.py  pipeline.py  serve.py
  viewer/                 built viewer, committed so `pip install` needs no Node

web/                      the browser half — viewer source
  src/Viewer.jsx          the UI: class picker, overlays, eigen-image strip
  src/scene.js            three.js: geometry, camera, picking, render loop

bundles/                  the two sample bundles (committed — 2 MB for both)
scripts/                  rebuilds those two from their source datasets
docs/bundle-format.md     the contract between the halves
```

`datasets/` is gitignored: raw images are yours, and the bundles built from them
are what's worth committing.

## Development

```bash
uv run pytest                   # analysis maths + bundle round-trip
cd web && npm install
npm run build                   # rebuilds src/latent_explorer/viewer/
```

To work on the viewer with hot reload, leave a `view` running in one terminal
and the dev server in another — Vite proxies `/bundles` to it, so you get real
data without rebuilding into the package on every edit:

```bash
uv run latent-explorer view bundles/cifar10 --no-open   # terminal 1
cd web && npm run dev                                   # terminal 2
```

`src/latent_explorer/viewer/` is a build artefact that is nonetheless
committed — that's what lets an installed copy serve the viewer with no Node
present. Rebuild it and commit the result whenever anything in `web/src/`
changes.

Rebuild the sample bundles (downloads ~950 MB the first time, cached under
`.cache/`):

```bash
uv run python scripts/build_builtin_bundles.py
```

## License

MIT.
