"""
train_cached.py
============================
Trains on PRECOMPUTED, frozen embeddings (see precompute_embeddings.py)
instead of running encoders live. No CaBuAr download, no augmentation, no
--head choice (linear_probe only, by design -- see cached_models.py) --
this is deliberately the fast, narrow-scope path once you've decided what
to actually run many experiments over.

Usage:
    # single-model baseline
    python train_cached.py --models croma --embeddings_dir embeddings/10m --patches_dir patches/10m

    # fusion: early (concat -> one probe) or late (sum of per-model probes)
    python train_cached.py --models croma smarties copernicusfm --fusion early \
        --embeddings_dir embeddings/10m --patches_dir patches/10m
"""

import os
import json
import math
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm

from configs import MODEL_SPECS, footprint_m_for, native_crop_px_for
from cached_loader import build_cached_loaders
from cached_models import build_cached_model
from train_utils import DiceBCELoss, compute_metrics, plot_history


def _to_device(views: dict, device: str) -> dict:
    return {name: emb.to(device) for name, emb in views.items()}


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device) -> float:
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
        scheduler.step()

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


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODEL_SPECS))
    parser.add_argument("--fusion", choices=("early", "late", "attn"), default=None,
                         help="required iff --models has more than one entry (attn: change head only)")
    parser.add_argument("--embeddings_dir", required=True, help="e.g. embeddings/10m, from precompute_embeddings.py")
    parser.add_argument("--patches_dir", required=True, help="e.g. patches/10m -- SAME one embeddings were computed from")
    parser.add_argument("--head", choices=("linear", "change"), default="linear",
                         help="linear: strict probe on [pre||post]; change: siamese temporal-fusion head")
    parser.add_argument("--hidden", type=int, default=128, help="change head only: hidden width")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5, help="fixed decision threshold (never tuned)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", default="models_cached")
    args = parser.parse_args()

    if args.fusion == "attn" and args.head != "change":
        parser.error("--fusion attn requires --head change")
    set_seed(args.seed)
    head_tag = "linear_probe" if args.head == "linear" else "change"
    run_name = "+".join(args.models) + f"_{head_tag}" + (f"_{args.fusion}" if args.fusion else "")
    os.makedirs(args.save_dir, exist_ok=True)

    footprint_m = footprint_m_for(args.models)
    out_size = native_crop_px_for(footprint_m)

    train_loader, val_loader, test_loader, train_ds = build_cached_loaders(
        args.embeddings_dir, args.patches_dir, args.models,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers,
        seed=args.seed,
    )
    embed_dims, grid_sizes = train_ds.embed_dims(), train_ds.grid_sizes()
    print(f"[{run_name}] cached embeddings from {args.embeddings_dir} "
          f"({len(train_loader.dataset)} train / {len(val_loader.dataset)} val / {len(test_loader.dataset)} test)")
    for name in args.models:
        print(f"  {name}: embed_dim={embed_dims[name]}  grid_size={grid_sizes[name]}")

    model = build_cached_model(embed_dims, grid_sizes, out_size, fusion_strategy=args.fusion, device=args.device,
                              head=args.head, hidden=args.hidden)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.head == "linear":
        expected = sum(embed_dims.values()) + 1
        assert trainable == expected, f"probe should have sum(C)+1={expected} params, got {trainable}"
    print(f"[{run_name}] ready | trainable params: {trainable:,}")

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps, warmup_steps = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup_steps if s < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps))),
    )

    save_path = os.path.join(args.save_dir, f"final_model_{run_name}.pth")
    plot_dir = os.path.join(args.save_dir, "plots", run_name)
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_precision": [], "val_recall": [], "val_iou": []}

    # Fixed schedule, fixed threshold, no early stopping, no checkpoint selection:
    # the model that gets tested is always the one after the last epoch.
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, args.device)
        val_metrics, _, _, val_loss = evaluate(model, val_loader, args.device, criterion=criterion,
                                               threshold=args.threshold)
        print(f"epoch {epoch + 1}/{args.epochs} | lr={scheduler.get_last_lr()[0]:.2e} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_f1={val_metrics['f1']:.4f} | val_iou={val_metrics['iou']:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        for k in ("f1", "precision", "recall", "iou"):
            history[f"val_{k}"].append(val_metrics[k])

    plot_history(history, plot_dir)
    with open(os.path.join(args.save_dir, f"history_{run_name}.json"), "w") as f:
        json.dump(history, f, indent=2)
    torch.save({"model": model.state_dict(), "epoch": args.epochs, "models": args.models,
                "fusion": args.fusion, "threshold": args.threshold}, save_path)

    test_metrics, _, _, test_loss = evaluate(model, test_loader, args.device, criterion=criterion,
                                              threshold=args.threshold)
    print(f"\n{'=' * 60}\nTEST RESULTS | {run_name}\n{'=' * 60}")
    print(f"  loss:      {test_loss:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  mIoU:      {test_metrics['iou']:.4f}")
    print(f"  precision: {test_metrics['precision']:.4f}")
    print(f"  recall:    {test_metrics['recall']:.4f}")
    print(f"  threshold: {args.threshold:.2f} (fixed)")

    with open(os.path.join(args.save_dir, f"results_{run_name}.json"), "w") as f:
        json.dump({"models": args.models, "head": args.head, "fusion": args.fusion, "seed": args.seed,
                    "trainable_params": trainable, "epochs": args.epochs, "lr": args.lr,
                    "val_metrics_final": {k: v[-1] for k, v in history.items() if k.startswith("val_") and k != "val_loss"},
                    "test_metrics": test_metrics, "test_loss": test_loss, "threshold": args.threshold},
                   f, indent=2)


if __name__ == "__main__":
    main()
