"""CLIP image encoding.

Produces L2-normalised feature vectors in CLIP's native dimensionality. Reducing
those to something drawable is `reduce.py`'s job — the two were tangled together
in the original `generate_embeddings`, which made it impossible to re-project an
existing encode without paying for inference again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .. import DEFAULT_MODEL, DEFAULT_PRETRAINED

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PRETRAINED",
    "Encoder",
    "encode_images",
    "load_encoder",
    "pick_device",
]


@dataclass
class Encoder:
    model: object
    preprocess: object
    device: str
    model_name: str
    pretrained: str

    @property
    def label(self) -> str:
        """What went into the bundle manifest, so a stale encode is identifiable."""
        return f"{self.model_name}/{self.pretrained}"


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon, for anyone running the downloadable tool on a Mac.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(
    model_name: str = DEFAULT_MODEL,
    pretrained: str = DEFAULT_PRETRAINED,
    device: str | None = None,
) -> Encoder:
    import open_clip

    device = pick_device(device)
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device).eval()
    return Encoder(
        model=model,
        preprocess=preprocess,
        device=device,
        model_name=model_name,
        pretrained=pretrained,
    )


def encode_images(
    paths: list[Path],
    encoder: Encoder,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode every image to a unit-norm CLIP vector. Returns (N, D) float32."""
    if not paths:
        raise ValueError("No images to encode.")

    batches: list[torch.Tensor] = []
    iterator = range(0, len(paths), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding", unit="batch")

    for start in iterator:
        chunk = paths[start : start + batch_size]
        tensors = []
        for path in chunk:
            # convert("RGB") matters: datasets in the wild carry greyscale and
            # RGBA images, and CLIP's transform expects three channels.
            with Image.open(path) as image:
                tensors.append(encoder.preprocess(image.convert("RGB")))

        batch = torch.stack(tensors).to(encoder.device)
        with torch.no_grad():
            features = encoder.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        batches.append(features.cpu())

    return torch.cat(batches).numpy().astype(np.float32)
