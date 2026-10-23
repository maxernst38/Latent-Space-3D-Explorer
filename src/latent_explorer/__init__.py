"""Encode image datasets with CLIP, project them to 3D, explore them in a browser.

The package is split along the line that matters for this project: everything
expensive and Python-only happens offline (`embed`, `analysis`, `bundle`), and
the result is a self-contained static bundle that the browser viewer renders.
Nothing in here draws anything.
"""

__version__ = "0.1.0"

BUNDLE_FORMAT_VERSION = 1

# Defined here rather than beside the code that uses them, so that `cli.py` can
# name them in --help text without importing embed/ — which would drag in all of
# PyTorch (and its ~20 OpenMP threads) just to print a usage string.
DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_e16"
