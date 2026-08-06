"""
fusion_v2/verify_galileo_shape.py
====================================
Runs ONE real forward pass through the actual Galileo checkpoint (not a
dummy encoder) and prints the shape at every step, so you can confirm or
refute the [B, H', W', T, 7, D] shape encoders.GalileoEncoder assumes for
Encoder.forward()'s return -- traced from single_file_galileo.py's source,
but never executed until now.

Needs: galileo/ cloned at the repo root (one level above fusion_v2/), and a
downloaded checkpoint folder (config.json + encoder.pt). No CaBuAr/dataset
needed -- uses a small random dummy tensor.

Usage:
    cd src   # or whatever you've named the fusion_v2 folder
    python verify_galileo_shape.py                              # uses <repo_root>/galileo/data/models/nano
    python verify_galileo_shape.py /absolute/path/to/checkpoint  # or point at a specific folder
"""

import sys

import torch

from configs import MODEL_SPECS
from encoders import GalileoEncoder


def main():
    checkpoint_folder = sys.argv[1] if len(sys.argv) > 1 else None   # None -> GalileoEncoder's own
                                                                       # repo-root-anchored default
    spec = MODEL_SPECS["galileo"]

    # small dummy crop -- 40px is 5 patches at patch_size=8, big enough to catch
    # indexing bugs, small enough to run fast/on CPU
    resize_px = 40
    print(f"Loading Galileo from {checkpoint_folder or '<default: <repo_root>/galileo/data/models/nano>'} ...")
    enc = GalileoEncoder(resize_px=resize_px, checkpoint_folder=checkpoint_folder)
    print(f"Loaded OK. embedding_size={enc.embed_dim}, patch_size={enc.patch_size}, "
          f"grid_size={enc.grid_size} (expected {resize_px // enc.patch_size})")

    B = 2
    pre = torch.randn(B, len(spec.bands), resize_px, resize_px)
    post = torch.randn(B, len(spec.bands), resize_px, resize_px)

    print("\n--- intermediate shapes (patching GalileoEncoder.forward to print) ---")
    s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = enc._build_masked_input(pre, post)
    print(f"s_t_x (masked input):  {tuple(s_t_x.shape)}   expected (B,H,W,T,13) = ({B},{resize_px},{resize_px},2,13)")
    print(f"s_t_m (mask):          {tuple(s_t_m.shape)}   expected (B,H,W,T,7)")

    with torch.no_grad():
        s_t_out, *_ = enc.model(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months,
            patch_size=enc.patch_size, input_resolution_m=enc.target_gsd_m,
        )
    print(f"\ns_t_out (Encoder.forward() return): {tuple(s_t_out.shape)}")
    print(f"  expected (B, H', W', T, 7, D) = ({B}, {enc.grid_size}, {enc.grid_size}, 2, 7, {enc.embed_dim})")
    if tuple(s_t_out.shape) == (B, enc.grid_size, enc.grid_size, 2, 7, enc.embed_dim):
        print("  MATCH -- the shape assumption in GalileoEncoder.forward() is correct.")
    else:
        print("  MISMATCH -- GalileoEncoder.forward()'s pooling step needs updating to match this real shape.")

    print("\n--- full GalileoEncoder.forward() ---")
    with torch.no_grad():
        out = enc(pre, post)
    print(f"final output: {tuple(out.shape)}   expected (B, D, H', W') = ({B}, {enc.embed_dim}, {enc.grid_size}, {enc.grid_size})")
    if tuple(out.shape) == (B, enc.embed_dim, enc.grid_size, enc.grid_size):
        print("  MATCH -- GalileoEncoder is ready to flip implemented=True in configs.py.")
    else:
        print("  MISMATCH -- do not flip implemented=True yet.")


if __name__ == "__main__":
    main()