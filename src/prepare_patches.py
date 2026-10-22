"""
prepare_patches.py
=============================
One-off preprocessing: tiles every CaBuAr pre/post/mask source image in each
split into a FULL grid of native crop_px x crop_px (20m/px) windows and
saves each split as {split}_patches.pt.

crop_px is now GROUP-specific (configs.GROUP_FOOTPRINT_M / CROP_GSD_M) since
different GSD groups use different shared footprints -- run this once per
group you plan to train on:

    python prepare_patches.py --group 10m --out_dir patches/10m

Patches are still model-agnostic WITHIN a group: band selection and
per-model resize happen later, at train time, in loader.py's PatchDataset
(driven by whatever --models you pass to train.py), so you don't need to
regenerate patches when swapping which model(s) within a group you're using.

Padding: edge windows that run past the true image bounds are zero-padded
on pre/post (same as loader.tile_source()) rather than dropped. The MASK is
padded with -1, marking those synthetic pixels as "no label" -- train.py's
loss and eval metrics both treat -1 as ignore_index.
"""

import os
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs import GROUP_FOOTPRINT_M, native_crop_px_for
from loader import FireDatasetV2, train_indices, val_indices, test_indices

MASK_IGNORE_VALUE = -1.0


def _tile_split(indices, crop_px: int, desc: str) -> dict:
    ds = FireDatasetV2(indices, model_names=[], augment=False)   # model_names unused by ._load()
    pre_list, post_list, mask_list = [], [], []
    src_idx_list, top_list, left_list = [], [], []

    pbar = tqdm(range(len(ds)), desc=desc, unit="img")
    for i in pbar:
        pre_full, post_full, mask_full = ds._load(i)
        _, H, W = pre_full.shape

        grid_h = -(-H // crop_px)
        grid_w = -(-W // crop_px)
        Hp, Wp = grid_h * crop_px, grid_w * crop_px
        pad_h, pad_w = Hp - H, Wp - W

        pre_p = F.pad(pre_full, (0, pad_w, 0, pad_h), value=0)
        post_p = F.pad(post_full, (0, pad_w, 0, pad_h), value=0)
        mask_p = F.pad(mask_full, (0, pad_w, 0, pad_h), value=MASK_IGNORE_VALUE)

        for gi in range(grid_h):
            for gj in range(grid_w):
                top, left = gi * crop_px, gj * crop_px
                pre_list.append(pre_p[:, top:top + crop_px, left:left + crop_px])
                post_list.append(post_p[:, top:top + crop_px, left:left + crop_px])
                mask_list.append(mask_p[:, top:top + crop_px, left:left + crop_px])
                src_idx_list.append(indices[i])
                top_list.append(top)
                left_list.append(left)

        pbar.set_postfix(patches=len(pre_list))

    return {
        "pre": torch.stack(pre_list),
        "post": torch.stack(post_list),
        "mask": torch.stack(mask_list),
        "src_idx": torch.tensor(src_idx_list, dtype=torch.long),
        "top": torch.tensor(top_list, dtype=torch.long),
        "left": torch.tensor(left_list, dtype=torch.long),
        "crop_px": crop_px,
        "mask_ignore_value": MASK_IGNORE_VALUE,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(GROUP_FOOTPRINT_M), required=True,
                         help="which GSD group's footprint to tile at (see configs.GROUP_FOOTPRINT_M)")
    parser.add_argument("--out_dir", default=None, help="default: patches/{group}")
    parser.add_argument("--crop_px", type=int, default=None,
                         help="override the group's derived crop_px (advanced -- normally leave unset)")
    args = parser.parse_args()

    crop_px = args.crop_px or native_crop_px_for(GROUP_FOOTPRINT_M[args.group])
    out_dir = args.out_dir or os.path.join("patches", args.group)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Group '{args.group}': footprint={GROUP_FOOTPRINT_M[args.group]}m -> crop_px={crop_px} "
          f"(at CaBuAr's native 20m/px)")

    for split, indices in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
        print(f"Tiling {split} split ({len(indices)} source images, crop_px={crop_px})...")
        data = _tile_split(indices, crop_px, split)
        n = data["pre"].shape[0]
        path = os.path.join(out_dir, f"{split}_patches.pt")
        torch.save(data, path)
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  saved {n} patches ({size_mb:.1f} MB) -> {path}\n")


if __name__ == "__main__":
    main()
