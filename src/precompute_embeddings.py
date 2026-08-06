"""
fusion_v2/precompute_embeddings.py
=====================================
Runs a frozen encoder ONCE over every patch in {train,val,test}_patches.pt
(from prepare_patches.py) and saves the resulting feature maps to disk, so
train_cached.py never has to run the (expensive, especially for galileo
base) encoder forward pass again.

DELIBERATE SCOPE DECISION: no augmentation. loader.py's flip/rot90
augmentation gives the live-encoder path a different pixel arrangement
every epoch; caching that exactly would mean either 16x the storage (one
cache entry per flip/rotation combo) or an approximation that's actually
wrong for Galileo specifically (absolute sin-cos position embeddings mean
flipping a CACHED output is not equivalent to encoding a flipped INPUT --
see the conversation this script came out of). Dropping augmentation
entirely is the standard, simpler choice for frozen-backbone linear
probing, and what was decided here. If you want augmentation back, you're
back to the live-encoder path (train.py --patches_dir), not this one.

Usage:
    python precompute_embeddings.py --patches_dir patches/10m --models croma smarties galileo \
        --out_dir embeddings/10m --croma_ckpt /path/to/CROMA_large.pt --galileo_ckpt_dir /path/to/base

Output layout (per model, per split):
    {out_dir}/{model}/{split}_embeddings.pt
        -- a single tensor (N, C, grid, grid), C/grid model-and-checkpoint-
           dependent, N matching {split}_patches.pt's patch count and order
           EXACTLY (index i here <-> index i in the original patches file,
           so masks are read from the original patches file at train time,
           not duplicated here).
    {out_dir}/{model}/meta.json
        -- embed_dim, grid_size, resize_px, footprint_m, source patches_dir,
           checkpoint path used -- so train_cached.py (and future-you) can
           sanity-check a cache actually matches what it's being paired with.
"""

import argparse
import gc
import json
import os

import torch
from tqdm import tqdm

from band_utils import select_bands
from configs import MODEL_SPECS, footprint_m_for, native_crop_px_for
from encoders import build_encoder
from resample import resize_to


def _default_croma_ckpt() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CROMA_large.pt")


def _already_done(out_path: str, expected_n: int) -> bool:
    """RESUME SUPPORT: if a previous run got killed partway (e.g. OOM), rerunning
    the same command should pick up where it left off, not recompute everything.
    A saved file only counts as done if its patch count matches the CURRENT
    patches file -- if patches were regenerated with a different crop_px/split
    size since this cache was written, we want the mismatch caught and
    recomputed, not silently trusted."""
    if not os.path.isfile(out_path):
        return False
    try:
        n = torch.load(out_path, map_location="cpu").shape[0]
    except Exception:
        return False   # corrupt/partial file (e.g. killed mid-torch.save) -- treat as not done
    return n == expected_n


def precompute_one(model_name: str, patches_dir: str, out_dir: str, device: str,
                    batch_size: int, croma_ckpt: str, galileo_ckpt_dir: str) -> None:
    spec = MODEL_SPECS[model_name]
    footprint_m = footprint_m_for([model_name])
    expected_crop_px = native_crop_px_for(footprint_m)
    resize_px = spec.resize_px_for(footprint_m)

    model_out_dir = os.path.join(out_dir, model_name)
    os.makedirs(model_out_dir, exist_ok=True)

    # check BEFORE loading the (potentially large, base-checkpoint-sized) encoder at
    # all -- if every split for this model is already done, skip loading it entirely
    split_ns = {}
    for split in ["train", "val", "test"]:
        data = torch.load(os.path.join(patches_dir, f"{split}_patches.pt"), map_location="cpu")
        if data["crop_px"] != expected_crop_px:
            raise ValueError(
                f"{split}_patches.pt's crop_px={data['crop_px']} doesn't match {model_name}'s expected "
                f"crop_px={expected_crop_px} (from footprint_m={footprint_m}) -- this patches_dir "
                f"wasn't generated for {model_name}'s GSD group. Re-run prepare_patches.py with the "
                f"matching --group, or double check --patches_dir."
            )
        split_ns[split] = data["pre"].shape[0]
        del data

    out_paths = {split: os.path.join(model_out_dir, f"{split}_embeddings.pt") for split in split_ns}
    if all(_already_done(out_paths[s], split_ns[s]) for s in split_ns):
        print(f"[{model_name}] all splits already cached in {model_out_dir} -- skipping (delete the "
              f"relevant *_embeddings.pt to force a recompute)")
        return

    kwargs = {}
    if model_name == "croma":
        kwargs["croma_ckpt"] = croma_ckpt
    if model_name == "galileo":
        kwargs["galileo_checkpoint_folder"] = galileo_ckpt_dir
    enc = build_encoder(model_name, resize_px, device, **kwargs)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False

    meta = {
        "model": model_name, "footprint_m": footprint_m, "resize_px": resize_px,
        "patches_dir": os.path.abspath(patches_dir),
        "checkpoint": croma_ckpt if model_name == "croma" else (galileo_ckpt_dir or "default"),
    }

    for split, n in split_ns.items():
        out_path = out_paths[split]
        if _already_done(out_path, n):
            print(f"[{model_name}] {split}: already cached ({n} patches) -- skipping")
            if split == "train":
                cached = torch.load(out_path, map_location="cpu")
                meta["embed_dim"], meta["grid_size"] = cached.shape[1], cached.shape[-1]
                del cached
            continue

        patches_path = os.path.join(patches_dir, f"{split}_patches.pt")
        data = torch.load(patches_path, map_location="cpu")
        pre_all = resize_to(select_bands(data["pre"], spec.bands), resize_px)
        post_all = resize_to(select_bands(data["post"], spec.bands), resize_px)
        del data   # the raw 12-band tensor isn't needed once band-selected+resized

        embed_dim, grid_size = None, None
        embeddings = None   # preallocated once we know embed_dim/grid_size from the first chunk --
                             # avoids holding a full chunk list AND the final concatenated tensor
                             # in memory at once (torch.cat would otherwise briefly double peak RAM)
        with torch.no_grad():
            starts = list(range(0, n, batch_size))
            for start in tqdm(starts, desc=f"[{model_name}] {split}", leave=False):
                pre_b = pre_all[start:start + batch_size].to(device)
                post_b = post_all[start:start + batch_size].to(device)
                feat = enc(pre_b, post_b).cpu()

                if embeddings is None:
                    embed_dim, grid_size = feat.shape[1], feat.shape[-1]
                    embeddings = torch.empty((n, embed_dim, grid_size, grid_size), dtype=feat.dtype)
                embeddings[start:start + feat.shape[0]] = feat

        del pre_all, post_all
        torch.save(embeddings, out_path)
        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"[{model_name}] {split}: saved {tuple(embeddings.shape)} ({size_mb:.1f} MB) -> {out_path}")

        if split == "train":
            meta["embed_dim"], meta["grid_size"] = embed_dim, grid_size
        del embeddings

    with open(os.path.join(model_out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # free this model's encoder (and any lingering GPU cache) before the next
    # model loads -- without this, memory from a big checkpoint (e.g. galileo
    # base) can stay resident while the next model's encoder is being built
    del enc
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patches_dir", required=True, help="e.g. patches/10m, from prepare_patches.py --group 10m")
    p.add_argument("--models", nargs="+", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--out_dir", required=True, help="e.g. embeddings/10m")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--croma_ckpt", default=_default_croma_ckpt())
    p.add_argument("--galileo_ckpt_dir", default=None)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        if not spec.implemented:
            raise NotImplementedError(f"'{model_name}' isn't wired up yet -- see configs.MODEL_SPECS['{model_name}'].")
        print(f"\n=== {model_name} ===")
        precompute_one(model_name, args.patches_dir, args.out_dir, args.device,
                        args.batch_size, args.croma_ckpt, args.galileo_ckpt_dir)


if __name__ == "__main__":
    main()