"""
fusion_v2/cached_loader.py
=============================
Dataset/loader for training on embeddings precomputed by
precompute_embeddings.py, instead of running encoders live.

Deliberately a SEPARATE module from loader.py: loader.py's module-level
code triggers an actual CaBuAr download from HF Hub as an import side
effect (see loader.py's own docstring) -- this module never needs the raw
CaBuAr dataset, only the cached embedding tensors and the ORIGINAL masks
saved by prepare_patches.py, so importing loader.py here would force an
unnecessary download every time train_cached.py runs.

No augmentation, no worker_init_fn needed -- __getitem__ is a pure,
deterministic index lookup (embeddings were precomputed once with no
randomness), so there's no cross-worker RNG-state concern the way
loader.py's flip/rotate augmentation had.
"""

import json
import os

import torch
from torch.utils.data import DataLoader, Dataset

BATCH_SIZE = 16


class CachedEmbeddingDataset(Dataset):
    def __init__(self, embeddings_dir: str, patches_dir: str, model_names: list, split: str):
        self.model_names = model_names
        self.embeddings = {}
        for name in model_names:
            emb_path = os.path.join(embeddings_dir, name, f"{split}_embeddings.pt")
            if not os.path.isfile(emb_path):
                raise FileNotFoundError(
                    f"No cached embeddings at {emb_path} -- run precompute_embeddings.py "
                    f"--models {name} first."
                )
            self.embeddings[name] = torch.load(emb_path)

        mask_path = os.path.join(patches_dir, f"{split}_patches.pt")
        self.mask = torch.load(mask_path)["mask"]

        n = self.mask.shape[0]
        for name, emb in self.embeddings.items():
            if emb.shape[0] != n:
                raise ValueError(
                    f"{name}'s {split}_embeddings.pt has {emb.shape[0]} entries but "
                    f"{mask_path} has {n} masks -- were these precomputed from the SAME "
                    f"{split}_patches.pt? Re-run precompute_embeddings.py against this exact "
                    f"--patches_dir if patches were regenerated since the cache was built."
                )

    def __len__(self) -> int:
        return self.mask.shape[0]

    def __getitem__(self, idx: int):
        views = {name: self.embeddings[name][idx] for name in self.model_names}
        return views, self.mask[idx]

    def embed_dims(self) -> dict:
        return {name: emb.shape[1] for name, emb in self.embeddings.items()}

    def grid_sizes(self) -> dict:
        return {name: emb.shape[-1] for name, emb in self.embeddings.items()}


def build_cached_loaders(embeddings_dir: str, patches_dir: str, model_names: list,
                          batch_size: int = BATCH_SIZE, eval_batch_size: "int | None" = None,
                          num_workers: int = 4):
    eval_batch_size = eval_batch_size or batch_size

    train_ds = CachedEmbeddingDataset(embeddings_dir, patches_dir, model_names, "train")
    val_ds = CachedEmbeddingDataset(embeddings_dir, patches_dir, model_names, "val")
    test_ds = CachedEmbeddingDataset(embeddings_dir, patches_dir, model_names, "test")

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True, persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader, train_ds