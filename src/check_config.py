"""
Prints the FULL encoder config from your Galileo checkpoint's config.json
(not just embedding_size, which galileo_embed_size.py already checked) --
specifically max_patch_size, which GalileoEncoder assumes is 8
(configs.MODEL_SPECS['galileo'].patch_size). If the checkpoint's actual
max_patch_size differs, every forward pass is silently running through
FlexiPatchEmbed's pseudo-inverse weight resampling instead of natively
trained weights -- degraded quality, not a crash, which would explain
weak-but-real signal without any obvious error.

Usage:
    python check_galileo_config.py                              # default nano checkpoint path
    python check_galileo_config.py /path/to/checkpoint/folder
"""

import json
import sys
from pathlib import Path

_DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "galileo" / "data" / "models" / "nano"

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_CHECKPOINT
with open(folder / "config.json") as f:
    config = json.load(f)

encoder_config = config["model"]["encoder"]
print(f"=== full encoder config from {folder / 'config.json'} ===")
for k, v in encoder_config.items():
    print(f"  {k}: {v}")

expected_patch_size = 8
actual = encoder_config.get("max_patch_size", "NOT SET -- falls back to Encoder.__init__'s default of 8")
print(f"\nconfigs.MODEL_SPECS['galileo'].patch_size (what we call forward() with): {expected_patch_size}")
print(f"checkpoint's actual max_patch_size: {actual}")
if str(actual) != str(expected_patch_size) and "NOT SET" not in str(actual):
    print("\n*** MISMATCH -- FlexiPatchEmbed IS resampling patch-embed weights on every forward pass. ***")
    print(f"    Fix: set patch_size={actual} for galileo in configs.MODEL_SPECS, not {expected_patch_size}.")
else:
    print("\nMatch -- patch embed uses native trained weights, no resampling. Not the cause.")