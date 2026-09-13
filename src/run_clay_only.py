"""
run_clay_only.py
====================
Standalone Clay-only pipeline: loads Clay's frozen weights, embeds every
patch in {train,val,test}_patches.pt (from prepare_patches.py --group 10m),
trains a linear-probe head, and reports val/test metrics.

Deliberately separate from precompute_embeddings.py / train.py /
train_cached.py -- no other model, no fusion, one file end to end -- so
Clay can be debugged/evaluated in complete isolation.

Prints embedding diagnostics (NaN/Inf, mean/std, per-position spatial std)
for every split BEFORE training starts. An all-zero F1/IoU after training is
the classic signature of degenerate frozen features -- either NaN (sigmoid
turns NaN into NaN, and `NaN > threshold` is False for every threshold, so
every prediction is silently "negative") or spatially-constant features
(nothing for a linear head to decode a mask from, no matter how long you
train it). This catches either one up front instead of after a full run.

Doesn't import loader.py -- that triggers a CaBuAr download from HF Hub as
an import side effect (see loader.py's own docstring). Reads patches/masks
straight from the {split}_patches.pt files prepare_patches.py already wrote.

Usage:
    python run_clay_only.py --patches_dir patches/10m \
        --clay_ckpt /path/to/clay-v1.5.ckpt --clay_metadata /path/to/metadata.yaml
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from band_utils import select_bands
from configs import MODEL_SPECS, footprint_m_for, native_crop_px_for
from encoders import build_encoder
from heads import build_head
from resample import resize_to
from train_utils import DiceBCELoss, compute_metrics, best_threshold, plot_history

MASK_IGNORE_VALUE = -1.0


# ============================================================
# DATA -- read straight from prepare_patches.py's output, no loader.py
# ============================================================

def _load_split(patches_dir: str, split: str, resize_px: int):
    data = torch.load(os.path.join(patches_dir, f"{split}_patches.pt"), map_location="cpu")
    spec = MODEL_SPECS["clay"]
    pre = resize_to(select_bands(data["pre"], spec.bands), resize_px)
    post = resize_to(select_bands(data["post"], spec.bands), resize_px)
    return pre, post, data["mask"]


@torch.no_grad()
def _embed_all(encoder, pre: torch.Tensor, post: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    n = pre.shape[0]
    embeddings = None
    for start in tqdm(range(0, n, batch_size), desc="embed", leave=False):
        pre_b = pre[start:start + batch_size].to(device)
        post_b = post[start:start + batch_size].to(device)
        feat = encoder(pre_b, post_b).cpu()
        if embeddings is None:
            embeddings = torch.empty((n, *feat.shape[1:]), dtype=feat.dtype)
        embeddings[start:start + feat.shape[0]] = feat
    return embeddings


def _print_diagnostics(name: str, emb: torch.Tensor) -> None:
    n_nan = torch.isnan(emb).sum().item()
    n_inf = torch.isinf(emb).sum().item()
    finite = torch.isfinite(emb)
    mean = emb[finite].mean().item() if finite.any() else float("nan")
    std = emb[finite].std().item() if finite.any() else float("nan")
    # per (sample, channel) std ACROSS the spatial grid, averaged -- ~0 means
    # every spatial position produced the same value (degenerate features,
    # nothing for a segmentation head to key on).
    spatial_std = emb.std(dim=(-2, -1)).mean().item()

    print(f"[{name}] shape={tuple(emb.shape)}  nan={n_nan}  inf={n_inf}  "
          f"mean={mean:.4f}  std={std:.4f}  spatial_std={spatial_std:.6f}")
    if n_nan or n_inf:
        print(f"  *** {name}: NaN/Inf in embeddings -- this alone explains an all-zero F1/IoU "
              f"(sigmoid(NaN) -> NaN -> every threshold comparison is False). The bug is inside "
              f"Clay's forward math (or the inputs feeding it), not the training loop.")
    if spatial_std < 1e-4:
        print(f"  *** {name}: embeddings are ~constant across space -- a linear head has nothing "
              f"to decode a segmentation mask from, regardless of how long you train it.")


# ============================================================
# TRAIN / EVAL (linear probe only, frozen encoder already embedded)
# ============================================================

def train_one_epoch(head, embs, masks, indices, batch_size, optimizer, criterion, device) -> float:
    head.train()
    total_loss = 0.0
    perm = indices[torch.randperm(len(indices))]
    for start in range(0, len(perm), batch_size):
        idx = perm[start:start + batch_size]
        feat = embs[idx].to(device)
        mask = masks[idx].to(device)

        optimizer.zero_grad()
        logits = head(feat, mask.shape[-1])
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)
    return total_loss / len(perm)


@torch.no_grad()
def evaluate(head, embs, masks, indices, batch_size, device, criterion, threshold: float = 0.5,
             ignore_index: float = MASK_IGNORE_VALUE):
    head.eval()
    all_probs, all_targets = [], []
    total_loss, n_batches = 0.0, 0
    for start in range(0, len(indices), batch_size):
        idx = indices[start:start + batch_size]
        feat = embs[idx].to(device)
        mask = masks[idx].to(device)

        logits = head(feat, mask.shape[-1])
        loss = criterion(logits, mask)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs.ravel())
        all_targets.append(mask.cpu().numpy().ravel())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    valid = targets != ignore_index
    probs, targets = probs[valid], targets[valid]
    preds = (probs > threshold).astype(np.uint8)
    return compute_metrics(preds, targets), probs, targets, total_loss / max(n_batches, 1)


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patches_dir", required=True, help="e.g. patches/10m, from prepare_patches.py --group 10m")
    p.add_argument("--clay_ckpt", required=True, help="path to clay-v1.5.ckpt (or whatever you renamed it to)")
    p.add_argument("--clay_metadata", required=True, help="path to Clay's metadata.yaml")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed_batch_size", type=int, default=8, help="batch size for the one-off encoder pass")
    p.add_argument("--batch_size", type=int, default=16, help="batch size for training the linear-probe head")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--save_dir", default="models_clay_only")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    spec = MODEL_SPECS["clay"]
    footprint_m = footprint_m_for(["clay"])
    resize_px = spec.resize_px_for(footprint_m)
    out_size = native_crop_px_for(footprint_m)
    print(f"[clay] footprint={footprint_m:.0f}m  resize_px={resize_px}  out_size={out_size}")

    print("Loading Clay encoder (frozen)...")
    encoder = build_encoder("clay", resize_px, args.device,
                             clay_ckpt=args.clay_ckpt, clay_metadata=args.clay_metadata)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    embs, masks = {}, {}
    for split in ["train", "val", "test"]:
        pre, post, mask = _load_split(args.patches_dir, split, resize_px)
        print(f"[{split}] embedding {pre.shape[0]} patches...")
        emb = _embed_all(encoder, pre, post, args.device, args.embed_batch_size)
        _print_diagnostics(split, emb)
        embs[split], masks[split] = emb, mask
        del pre, post

    # frozen encoder is no longer needed once every split is embedded -- free it
    # before training the head (same cleanup precompute_embeddings.py does)
    del encoder
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    head = build_head("linear_probe", embs["train"].shape[1]).to(args.device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_idx = torch.arange(embs["train"].shape[0])
    val_idx = torch.arange(embs["val"].shape[0])
    test_idx = torch.arange(embs["test"].shape[0])

    best_f1, best_epoch, patience_ct = 0.0, -1, 0
    save_path = os.path.join(args.save_dir, "best_model_clay.pth")
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_f1_best_thr": [],
               "val_precision": [], "val_recall": [], "val_iou": [], "threshold": []}

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(head, embs["train"], masks["train"], train_idx,
                                      args.batch_size, optimizer, criterion, args.device)
        val_metrics, val_probs, val_targets, val_loss = evaluate(
            head, embs["val"], masks["val"], val_idx, args.batch_size, args.device, criterion)
        thr, thr_f1 = best_threshold(val_probs, val_targets.astype(np.uint8))

        print(f"epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
              f"val_f1(0.5)={val_metrics['f1']:.4f} | val_f1(best_thr={thr:.2f})={thr_f1:.4f} | "
              f"val_iou={val_metrics['iou']:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_metrics["f1"])
        history["val_f1_best_thr"].append(thr_f1)
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_iou"].append(val_metrics["iou"])
        history["threshold"].append(thr)

        plot_history(history, os.path.join(args.save_dir, "plots"))
        with open(os.path.join(args.save_dir, "history_clay.json"), "w") as f:
            json.dump(history, f, indent=2)

        if thr_f1 > best_f1:
            best_f1, best_epoch, patience_ct = thr_f1, epoch, 0
            torch.save({"model": head.state_dict(), "threshold": thr, "epoch": epoch}, save_path)
        else:
            patience_ct += 1
            if patience_ct >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (best epoch {best_epoch + 1}, val_f1={best_f1:.4f})")
                break

    ckpt = torch.load(save_path, map_location=args.device)
    head.load_state_dict(ckpt["model"])
    test_metrics, _, _, test_loss = evaluate(head, embs["test"], masks["test"], test_idx,
                                              args.batch_size, args.device, criterion,
                                              threshold=ckpt["threshold"])
    print(f"\n{'=' * 60}\nCLAY-ONLY TEST RESULTS\n{'=' * 60}")
    print(f"  loss:      {test_loss:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  mIoU:      {test_metrics['iou']:.4f}")
    print(f"  precision: {test_metrics['precision']:.4f}")
    print(f"  recall:    {test_metrics['recall']:.4f}")
    print(f"  threshold: {ckpt['threshold']:.2f} (best on val)")

    with open(os.path.join(args.save_dir, "results_clay.json"), "w") as f:
        json.dump({"model": "clay", "head": "linear_probe", "best_epoch": best_epoch, "val_f1": best_f1,
                    "test_metrics": test_metrics, "test_loss": test_loss, "threshold": ckpt["threshold"]},
                   f, indent=2)


if __name__ == "__main__":
    main()
