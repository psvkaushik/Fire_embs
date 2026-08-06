"""
fusion_v2/galileo_embed_size.py
==================================
Prints the embedding size for a Galileo checkpoint folder. Reads it straight
out of config.json (the same file Encoder.load_from_folder() uses to build
the model) -- no torch import, no weight loading, instant.

Usage:
    python galileo_embed_size.py                              # <repo_root>/galileo/data/models/nano
    python galileo_embed_size.py /absolute/path/to/checkpoint  # or point at a specific folder
"""

import json
import sys
from pathlib import Path

# galileo/ is a SIBLING of this file's own folder (<repo_root>/galileo, <repo_root>/src or fusion_v2/),
# not nested inside it -- anchor to this file's own location so it works regardless of cwd or what
# this folder is named, same fix as encoders.GalileoEncoder's checkpoint_folder default.
_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "galileo" / "data" / "models" / "nano"


def galileo_embedding_size(checkpoint_folder) -> int:
    config_path = Path(checkpoint_folder) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    return config["model"]["encoder"]["embedding_size"]


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_CHECKPOINT
    size = galileo_embedding_size(folder)
    print(f"{folder}: embedding_size = {size}")