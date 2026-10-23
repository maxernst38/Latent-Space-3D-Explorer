"""CLIP encoding and dimensionality reduction.

Deliberately empty of re-exports. Importing `clip` here would mean that anything
touching `embed.reduce` — including the bundle writer, and therefore the whole
CLI — paid for a full PyTorch import. Import the submodule you actually need:

    from latent_explorer.embed.reduce import project   # numpy + sklearn
    from latent_explorer.embed.clip import load_encoder  # pulls in torch
"""
