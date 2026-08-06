"""
fusion_v2/loader.py
====================
Same crop/augment/tile machinery as v1, generalized twice over:
  1. any subset of configs.MODEL_SPECS can be requested (build_loaders(model_names=...))
  2. the crop pulled from CaBuAr, and the size each model gets resampled to,
     are now derived from configs.footprint_m_for(model_names) -- the RUN's
     shared ground footprint (per GSD group) -- instead of one hardcoded
     168px crop for everything. A 10m-group run crops 112px (=2240m/20m)
     from CaBuAr; a 30m-group run crops 336px (=6720m/20m). See configs.py
     for why those numbers.

Per model, a "view" is: band_utils.select_bands (pick this model's bands)
-> resample.resize_to (resize the shared native crop to THIS model's
resize_px for this run's footprint, from ModelSpec.resize_px_for()).
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import train_test_split

from configs import (MODEL_SPECS, CROP_GSD_M, N_BANDS, CABUAR_S2_MEAN, CABUAR_S2_STD,
                      footprint_m_for, native_crop_px_for)
from band_utils import select_bands
from resample import resize_to

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def _worker_init_fn(worker_id: int) -> None:
    """DataLoader workers are forked from the main process, so without this
    they'd all inherit the SAME `random`/`np.random` state and produce
    CORRELATED augmentations (same flips/rotations across workers within a
    batch) -- silently biasing what the model sees. torch's own RNG IS
    reseeded per-worker automatically (that's what torch.initial_seed()
    below reads back), but random/np.random are not; derive distinct seeds
    from it so each worker's augmentation stream is actually independent."""
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)

# ============================================================
# CONSTANTS
# ============================================================

N_SAMPLES      = 424
TEST_START_IDX = 356
BATCH_SIZE     = 4

# ============================================================
# DATASET-LEVEL NORMALIZATION (12 bands, CaBuAr channel order)
# Values live in configs.py (CABUAR_S2_MEAN/STD, plain lists) so encoders.py
# can reference the same numbers without importing this module -- this
# module's load does an actual CaBuAr download from HF Hub as a side effect
# of import, which you don't want triggered just to read two constants.
# ============================================================

S2_MEAN = torch.tensor(CABUAR_S2_MEAN).view(N_BANDS, 1, 1)
S2_STD = torch.tensor(CABUAR_S2_STD).view(N_BANDS, 1, 1)

# ============================================================
# LOAD RAW DATASET
# ============================================================

print("Loading CaBuAr dataset...")

dataset = load_dataset("DarthReca/california_burned_areas", name="pre-post-fire", trust_remote_code=True)

full_data = concatenate_datasets([
    dataset['0'], dataset['1'], dataset['2'], dataset['3'], dataset['4'], dataset['chabud']
])

assert len(full_data) == N_SAMPLES, f"Expected {N_SAMPLES} samples, got {len(full_data)}"
print(f"Dataset loaded ✅  ({len(full_data)} samples)")

# ============================================================
# SPLITS
# ============================================================

all_idx      = np.arange(N_SAMPLES)
test_indices = [int(i) for i in all_idx[TEST_START_IDX:]]
tv_indices   = all_idx[:TEST_START_IDX]

train_indices, val_indices = train_test_split(tv_indices, test_size=0.20, random_state=SEED, shuffle=True)
train_indices = [int(i) for i in train_indices]
val_indices   = [int(i) for i in val_indices]

print(f"Train={len(train_indices)}  Val={len(val_indices)}  Test={len(test_indices)}")

# ============================================================
# PER-MODEL VIEW CONSTRUCTION
# ============================================================

def make_model_views(pre_crop: torch.Tensor, post_crop: torch.Tensor, model_names: list,
                      footprint_m: float) -> dict:
    """pre_crop/post_crop: (12, crop_px, crop_px) at CaBuAr's native 20m/px,
    crop_px = native_crop_px_for(footprint_m). Returns {model_name: (pre, post)},
    each band-selected then resized to ModelSpec.resize_px_for(footprint_m)."""
    views = {}
    for name in model_names:
        spec = MODEL_SPECS[name]
        pre_b, post_b = select_bands(pre_crop, spec.bands), select_bands(post_crop, spec.bands)
        size_px = spec.resize_px_for(footprint_m)
        views[name] = (resize_to(pre_b, size_px), resize_to(post_b, size_px))
    return views


# ============================================================
# AUGMENTATION
# ============================================================

def _flip_views(views: dict, mask: torch.Tensor, dim: int):
    views = {name: (torch.flip(p, [dim]), torch.flip(q, [dim])) for name, (p, q) in views.items()}
    return views, torch.flip(mask, [dim])


def _rot90_views(views: dict, mask: torch.Tensor, k: int):
    views = {name: (torch.rot90(p, k, [-2, -1]), torch.rot90(q, k, [-2, -1])) for name, (p, q) in views.items()}
    return views, torch.rot90(mask, k, [-2, -1])


# ============================================================
# DATASET
# ============================================================

class FireDatasetV2(Dataset):
    """
    Training (augment=True):
        returns (views, mask) where views = {model_name: (pre, post)},
        mask shaped (1, crop_px, crop_px) at CaBuAr's native 20m/px --
        crop_px determined by this run's footprint (see configs.py).

    Validation/Test (augment=False):
        returns (pre, post, mask) FULL-SIZE (12, H, W) source tensors --
        tiling into per-model crops happens later via tile_source().
    """

    def __init__(self, indices, model_names, augment=False):
        self.indices = indices
        self.model_names = model_names
        self.augment = augment
        self.footprint_m = footprint_m_for(model_names) if model_names else None
        self.crop_px = native_crop_px_for(self.footprint_m) if self.footprint_m else None

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        return (x - S2_MEAN) / (S2_STD + 1e-6)

    def _load(self, idx):
        s = full_data[self.indices[idx]]

        pre = torch.tensor(np.array(s["pre_fire"], dtype=np.float32)).permute(2, 0, 1)
        post = torch.tensor(np.array(s["post_fire"], dtype=np.float32)).permute(2, 0, 1)
        pre, post = self.normalize(pre), self.normalize(post)

        mask = np.array(s["mask"], dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[..., 0] if mask.shape[-1] == 1 else mask[0]
        mask = (mask > 0).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return pre, post, mask

    def __getitem__(self, idx):
        pre, post, mask = self._load(idx)

        if not self.augment:
            return pre, post, mask

        _, H, W = pre.shape
        cpx = self.crop_px
        top = random.randint(0, H - cpx)
        left = random.randint(0, W - cpx)

        pre_c = pre[:, top:top + cpx, left:left + cpx]
        post_c = post[:, top:top + cpx, left:left + cpx]
        mask_c = mask[:, top:top + cpx, left:left + cpx]

        views = make_model_views(pre_c, post_c, self.model_names, self.footprint_m)

        if random.random() > 0.5:
            views, mask_c = _flip_views(views, mask_c, -1)
        if random.random() > 0.5:
            views, mask_c = _flip_views(views, mask_c, -2)
        k = random.randint(0, 3)
        if k > 0:
            views, mask_c = _rot90_views(views, mask_c, k)

        return views, mask_c


def _make_collate_train(model_names):
    def _collate_train(batch):
        views_out = {
            name: (
                torch.stack([b[0][name][0] for b in batch]),
                torch.stack([b[0][name][1] for b in batch]),
            )
            for name in model_names
        }
        mask_out = torch.stack([b[1] for b in batch])
        return views_out, mask_out
    return _collate_train


def build_loaders(model_names: list, batch_size: int = BATCH_SIZE, num_workers: int = 4):
    train_ds = FireDatasetV2(train_indices, model_names, augment=True)
    val_ds   = FireDatasetV2(val_indices,   model_names, augment=False)
    test_ds  = FireDatasetV2(test_indices,  model_names, augment=False)

    collate = _make_collate_train(model_names)
    # persistent_workers avoids re-forking (and re-running each worker's CaBuAr-adjacent
    # imports) every epoch; only valid when num_workers>0, hence the guard.
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=persistent, prefetch_factor=(4 if persistent else None),
    )
    # val/test are unaugmented (deterministic) and batch_size=1 (full-image tiling happens
    # downstream in tile_source), so worker seeding doesn't matter here -- still parallelized
    # for the same "don't block the GPU on CPU-side tensor prep" reason.
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# FULL-IMAGE TILING (VALIDATION / TEST)
# ============================================================

def tile_source(pre_full: torch.Tensor, post_full: torch.Tensor, model_names: list):
    """Cover a full-size (12, H, W) source image with a grid of native
    crop_px x crop_px windows (crop_px derived from this run's footprint),
    zero-padded at the bottom/right edge to the next multiple of crop_px,
    and build the requested model views for every tile.

    Returns:
        tiles: list of (views_dict, top, left) -- top/left index into the
               PADDED canvas, in native-crop-px units, for stitching logits back.
        H, W:  the ORIGINAL (unpadded) source height/width.
        Hp, Wp: the padded canvas height/width (multiples of crop_px).
        crop_px: this run's native crop size (needed by the caller to know
                 the stitching stride).
    """
    footprint_m = footprint_m_for(model_names)
    crop_px = native_crop_px_for(footprint_m)

    _, H, W = pre_full.shape
    grid_h = -(-H // crop_px)
    grid_w = -(-W // crop_px)
    Hp, Wp = grid_h * crop_px, grid_w * crop_px
    pad_h, pad_w = Hp - H, Wp - W

    pre_p = F.pad(pre_full, (0, pad_w, 0, pad_h), value=0)
    post_p = F.pad(post_full, (0, pad_w, 0, pad_h), value=0)

    tiles = []
    for i in range(grid_h):
        for j in range(grid_w):
            top, left = i * crop_px, j * crop_px
            pre_c = pre_p[:, top:top + crop_px, left:left + crop_px]
            post_c = post_p[:, top:top + crop_px, left:left + crop_px]
            views = make_model_views(pre_c, post_c, model_names, footprint_m)
            tiles.append((views, top, left))

    return tiles, H, W, Hp, Wp, crop_px


# ============================================================
# PRE-TILED PATCH DATASET
#
# Patches are stored per-GROUP now (crop_px depends on footprint, which
# depends on GSD group) -- prepare_patches.py writes {group}/{split}_patches.pt.
# Still model-agnostic WITHIN a group: band selection + per-model resize
# happens here at __getitem__ time, driven by model_names.
# ============================================================

class PatchDataset(Dataset):
    def __init__(self, patches_path: str, model_names: list, augment: bool = False):
        data = torch.load(patches_path)
        self.pre = data["pre"]
        self.post = data["post"]
        self.mask = data["mask"]
        self.model_names = model_names
        self.footprint_m = footprint_m_for(model_names)
        self.augment = augment

    def __len__(self):
        return self.pre.shape[0]

    def __getitem__(self, idx):
        pre, post, mask = self.pre[idx], self.post[idx], self.mask[idx]
        views = make_model_views(pre, post, self.model_names, self.footprint_m)

        if not self.augment:
            return views, mask

        if random.random() > 0.5:
            views, mask = _flip_views(views, mask, -1)
        if random.random() > 0.5:
            views, mask = _flip_views(views, mask, -2)
        k = random.randint(0, 3)
        if k > 0:
            views, mask = _rot90_views(views, mask, k)

        return views, mask


def build_patch_loaders(patches_dir: str, model_names: list, batch_size: int = BATCH_SIZE,
                         eval_batch_size: "int | None" = None, num_workers: int = 4):
    """patches_dir should be the GROUP-specific directory written by
    `prepare_patches.py --group 10m` (or 30m) -- i.e. containing
    train_patches.pt / val_patches.pt / test_patches.pt for models_names'
    shared GSD group."""
    eval_batch_size = eval_batch_size or batch_size
    collate = _make_collate_train(model_names)
    persistent = num_workers > 0

    train_ds = PatchDataset(os.path.join(patches_dir, "train_patches.pt"), model_names, augment=True)
    val_ds   = PatchDataset(os.path.join(patches_dir, "val_patches.pt"),   model_names, augment=False)
    test_ds  = PatchDataset(os.path.join(patches_dir, "test_patches.pt"),  model_names, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=persistent, prefetch_factor=(4 if persistent else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, collate_fn=collate, persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader