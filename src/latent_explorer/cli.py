"""Command line entry point.

    latent-explorer build datasets/my-set -o bundles/my-set
    latent-explorer view bundles/my-set
    latent-explorer inspect bundles/my-set

argparse rather than click or typer: three subcommands do not justify a
dependency in a tool whose install is already a multi-gigabyte PyTorch download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Only torch-free imports at module scope. `build` pulls in the heavy modules
# inside _run_build, so `view` and `inspect` start instantly instead of paying
# a three-and-a-half second PyTorch import they never use.
from . import DEFAULT_MODEL, DEFAULT_PRETRAINED, __version__, datasets
from .bundle.thumbnails import DEFAULT_TILE


def _add_build(subparsers) -> None:
    p = subparsers.add_parser(
        "build", help="Encode a dataset and write a viewable bundle."
    )
    p.add_argument("dataset", type=Path, help="Dataset root directory.")
    p.add_argument("-o", "--out", type=Path, required=True, help="Bundle output dir.")
    p.add_argument(
        "-f", "--format", choices=datasets.FORMATS, default=None,
        help="Dataset layout. Autodetected when omitted.",
    )
    p.add_argument("--split", default=None, help="Split to use (YOLO datasets).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="open_clip model name.")
    p.add_argument("--pretrained", default=DEFAULT_PRETRAINED, help="Pretrained tag.")
    p.add_argument(
        "--device", default=None, help="cuda, mps or cpu. Autodetected when omitted."
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--tile", type=int, default=DEFAULT_TILE, help="Thumbnail px.")
    p.add_argument("--limit", type=int, default=None, help="Sample at most N images.")
    p.add_argument("--no-thumbnails", action="store_true")
    p.add_argument("--id", default=None, help="Bundle id. Defaults to the output dir.")
    p.add_argument("--title", default=None, help="Display name shown in the viewer.")
    p.add_argument(
        "--cache", type=Path, default=None,
        help="Reuse/store the CLIP feature matrix here to skip re-encoding.",
    )


def _add_view(subparsers) -> None:
    p = subparsers.add_parser("view", help="Open one or more bundles in the viewer.")
    p.add_argument("bundles", type=Path, nargs="+")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--no-open", action="store_true", help="Don't launch a browser.")


def _add_inspect(subparsers) -> None:
    p = subparsers.add_parser("inspect", help="Summarise a bundle without opening it.")
    p.add_argument("bundle", type=Path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="latent-explorer",
        description="Encode image datasets with CLIP and explore them in 3D.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_build(subparsers)
    _add_view(subparsers)
    _add_inspect(subparsers)

    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            return _run_build(args)
        if args.command == "view":
            return _run_view(args)
        if args.command == "inspect":
            return _run_inspect(args)
    except (ValueError, FileNotFoundError) as exc:
        # These are the expected failures — a missing directory, an unrecognised
        # layout, too few images. A traceback would bury the actual message.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


def _run_build(args) -> int:
    from .pipeline import build

    manifest = build(
        dataset_root=args.dataset,
        out_dir=args.out,
        fmt=args.format,
        split=args.split,
        model_name=args.model,
        pretrained=args.pretrained,
        device=args.device,
        batch_size=args.batch_size,
        tile=args.tile,
        limit=args.limit,
        thumbnails=not args.no_thumbnails,
        features_cache=args.cache,
        bundle_id=args.id,
        title=args.title,
    )

    variance = manifest["projection"]["totalExplained"]
    print(
        f"\nWrote {manifest['count']} points across {len(manifest['classes'])} classes "
        f"to {args.out}"
    )
    print(f"  3 components capture {variance:.1%} of the variance")
    print(f"  view it with: latent-explorer view {args.out}")
    return 0


def _run_view(args) -> int:
    from .serve import serve

    serve(args.bundles, host=args.host, port=args.port, launch_browser=not args.no_open)
    return 0


def _run_inspect(args) -> int:
    from .bundle import read_manifest

    manifest = read_manifest(args.bundle)
    print(f"{manifest['name']}  ({manifest['id']})")
    print(f"  points   {manifest['count']}")
    print(f"  model    {manifest['model']}")
    print(f"  variance {manifest['projection']['totalExplained']:.1%} in 3 components")
    print(f"  built    {manifest['createdAt']}")

    classes = sorted(manifest["classes"], key=lambda c: -c["count"])
    print(f"  classes  {len(classes)}")
    for entry in classes[:10]:
        hull = "hull" if "hull" in entry else "    "
        print(f"    {entry['count']:>6}  {hull}  {entry['name']}")
    if len(classes) > 10:
        print(f"    … and {len(classes) - 10} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
