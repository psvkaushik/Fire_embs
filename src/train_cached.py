"""
fusion_v2/train_cached.py
============================
Trains on PRECOMPUTED, frozen embeddings (see precompute_embeddings.py)
instead of running encoders live. No CaBuAr download, no augmentation, no
--head choice (linear_probe only, by design -- see cached_models.py) --
this is deliberately the fast, narrow-scope path once you've decided what
to actually run many experiments over.

Usage:
    # single-model baseline
    python train_cached.py --models croma --embeddings_dir embeddings/10m --patches_dir patches/10m

    # fusion (same 3 strategies as train.py, same 10m-group membership rules)
    python train_cached.py --models croma smarties galileo --fusion early \
        --embeddings_dir embeddings/10m --patches_dir patches/10m
"""

import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

from configs import FUSION_STRATEGIES, MODEL_SPECS, footprint_m_for, native_crop_px_for
from cached_loader import build_cached_loaders
from cached_models import build_cached_model
from train_utils import DiceBCELoss, compute_metrics, best_threshold, plot_history


def _to_device(views: dict, device: str) -> dict:
    return {name: emb.to(device) for name, emb in views.items()}


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for views, mask in tqdm(loader, desc="train", leave=False):
        batch = _to_device(views, device)
        mask = mask.to(device)

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * mask.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None, threshold: float = 0.5, ignore_index: float = -1.0):
    model.eval()
    all_probs, all_targets = [], []
    total_loss, n_batches = 0.0, 0

    for views, mask in tqdm(loader, desc="eval", leave=False):
        batch = _to_device(views, device)
        mask = mask.to(device)

        logits = model(batch)
        if criterion is not None:
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
    val_loss = total_loss / n_batches if n_batches else None
    return compute_metrics(preds, targets), probs, targets, val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODEL_SPECS))
    parser.add_argument("--fusion", choices=FUSION_STRATEGIES, default=None,
                         help="required iff --models has more than one entry")
    parser.add_argument("--embeddings_dir", required=True, help="e.g. embeddings/10m, from precompute_embeddings.py")
    parser.add_argument("--patches_dir", required=True, help="e.g. patches/10m -- SAME one embeddings were computed from")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--proj_dim", type=int, default=256, help="early fusion: projected channel dim")
    parser.add_argument("--dec_dim", type=int, default=128, help="sep_dec fusion: per-branch decoder channel dim")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="models_cached")
    args = parser.parse_args()

    run_name = "+".join(args.models) + "_linear_probe" + (f"_{args.fusion}" if args.fusion else "")
    os.makedirs(args.save_dir, exist_ok=True)

    footprint_m = footprint_m_for(args.models)
    out_size = native_crop_px_for(footprint_m)

    train_loader, val_loader, test_loader, train_ds = build_cached_loaders(
        args.embeddings_dir, args.patches_dir, args.models,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
    )
    embed_dims, grid_sizes = train_ds.embed_dims(), train_ds.grid_sizes()
    print(f"[{run_name}] cached embeddings from {args.embeddings_dir} "
          f"({len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test)")
    for name in args.models:
        print(f"  {name}: embed_dim={embed_dims[name]}  grid_size={grid_sizes[name]}")

    model = build_cached_model(
        embed_dims, grid_sizes, out_size, fusion_strategy=args.fusion,
        proj_dim=args.proj_dim, dec_dim=args.dec_dim, device=args.device,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{run_name}] ready | trainable params: {trainable:,} (no frozen encoder -- it's not even loaded here)")

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1, best_epoch, patience_ct = 0.0, -1, 0
    save_path = os.path.join(args.save_dir, f"best_model_{run_name}.pth")
    plot_dir = os.path.join(args.save_dir, "plots", run_name)
    history = {
        "train_loss": [], "val_loss": [],
        "val_f1": [], "val_f1_best_thr": [], "val_precision": [], "val_recall": [], "val_iou": [],
        "threshold": [],
    }

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)

        val_metrics, val_probs, val_targets, val_loss = evaluate(model, val_loader, args.device, criterion=criterion)
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

        plot_history(history, plot_dir)
        with open(os.path.join(args.save_dir, f"history_{run_name}.json"), "w") as f:
            json.dump(history, f, indent=2)

        if thr_f1 > best_f1:
            best_f1, best_epoch, patience_ct = thr_f1, epoch, 0
            torch.save({"model": model.state_dict(), "threshold": thr, "epoch": epoch,
                        "models": args.models, "fusion": args.fusion}, save_path)
        else:
            patience_ct += 1
            if patience_ct >= args.patience:
                print(f"Early stopping at epoch {epoch + 1} (best epoch {best_epoch + 1}, val_f1={best_f1:.4f})")
                break

    ckpt = torch.load(save_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    test_metrics, _, _, test_loss = evaluate(model, test_loader, args.device, criterion=criterion,
                                              threshold=ckpt["threshold"])
    print(f"\n{'=' * 60}\nTEST RESULTS | {run_name}\n{'=' * 60}")
    print(f"  loss:      {test_loss:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  mIoU:      {test_metrics['iou']:.4f}")
    print(f"  precision: {test_metrics['precision']:.4f}")
    print(f"  recall:    {test_metrics['recall']:.4f}")
    print(f"  threshold: {ckpt['threshold']:.2f} (best on val)")

    with open(os.path.join(args.save_dir, f"results_{run_name}.json"), "w") as f:
        json.dump({"models": args.models, "head": "linear_probe", "fusion": args.fusion,
                    "best_epoch": best_epoch, "val_f1": best_f1,
                    "test_metrics": test_metrics, "test_loss": test_loss, "threshold": ckpt["threshold"]},
                   f, indent=2)


if __name__ == "__main__":
    main()