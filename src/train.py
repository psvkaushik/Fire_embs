# """
# fusion_v2/train.py
# ====================
# Trains ANY single model or ANY fusion combination on CaBuAr wildfire
# segmentation.

# Usage:
#     # single-model baselines
#     python train.py --models croma --head linear_probe
#     python train.py --models croma --head upernet
#     python train.py --models prithvi --head upernet --unfreeze_encoders

#     # fusion within the 10m group (croma/smarties/galileo) -- 30m group
#     # (prithvi) currently has just one member, nothing to fuse there yet
#     python train.py --models croma smarties galileo --head upernet --fusion early
#     python train.py --models croma smarties galileo --head linear_probe --fusion sep_dec
#     python train.py --models croma smarties galileo --head upernet --fusion late
# """

# import os
# import json
# import argparse

# import numpy as np
# import torch
# import torch.nn as nn
# from tqdm import tqdm
# from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from configs import MODEL_SPECS, HEAD_TYPES, FUSION_STRATEGIES
# from loader import build_loaders, build_patch_loaders, tile_source
# from models import build_model


# # ============================================================
# # LOSS  (unchanged from v1)
# # ============================================================

# class DiceBCELoss(nn.Module):
#     def __init__(self, ignore_index: float = -1.0):
#         super().__init__()
#         self.ignore_index = ignore_index
#         self.bce = nn.BCEWithLogitsLoss(reduction="none")

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         valid = (targets != self.ignore_index).float()
#         targets = torch.where(valid.bool(), targets, torch.zeros_like(targets))

#         bce_raw = self.bce(logits, targets)
#         bce = (bce_raw * valid).sum() / valid.sum().clamp(min=1.0)

#         probs = torch.sigmoid(logits).flatten(1) * valid.flatten(1)
#         t = targets.flatten(1)
#         inter = (probs * t).sum(dim=1)
#         union = probs.sum(dim=1) + t.sum(dim=1)
#         dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
#         return 0.5 * bce + 0.5 * dice.mean()


# # ============================================================
# # METRICS  (unchanged from v1)
# # ============================================================

# def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
#     preds = preds.astype(np.uint8).ravel()
#     targets = targets.astype(np.uint8).ravel()
#     return {
#         "f1": f1_score(targets, preds, zero_division=0),
#         "precision": precision_score(targets, preds, zero_division=0),
#         "recall": recall_score(targets, preds, zero_division=0),
#         "iou": jaccard_score(targets, preds, zero_division=0),
#     }


# def best_threshold(probs: np.ndarray, targets: np.ndarray):
#     best_f1, best_t = 0.0, 0.5
#     for t in np.linspace(0.1, 0.9, 17):
#         f1 = f1_score(targets, (probs > t).astype(np.uint8), zero_division=0)
#         if f1 > best_f1:
#             best_f1, best_t = f1, t
#     return float(best_t), float(best_f1)


# # ============================================================
# # HISTORY / PLOTTING  (unchanged from v1)
# # ============================================================

# def plot_history(history: dict, plot_dir: str) -> None:
#     os.makedirs(plot_dir, exist_ok=True)
#     epochs = np.arange(1, len(history["train_loss"]) + 1)

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(epochs, history["train_loss"], label="train loss")
#     ax.plot(epochs, history["val_loss"], label="val loss")
#     ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training / validation loss")
#     ax.legend(); ax.grid(alpha=0.3)
#     fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "loss_curve.png")); plt.close(fig)

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(epochs, history["val_f1"], label="val F1 (thr=0.5)")
#     ax.plot(epochs, history["val_f1_best_thr"], label="val F1 (best thr)", linestyle="--")
#     ax.set_xlabel("epoch"); ax.set_ylabel("F1"); ax.set_ylim(0, 1); ax.set_title("Validation F1")
#     ax.legend(); ax.grid(alpha=0.3)
#     fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "f1_curve.png")); plt.close(fig)

#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(epochs, history["val_iou"], label="val IoU")
#     ax.set_xlabel("epoch"); ax.set_ylabel("IoU"); ax.set_ylim(0, 1); ax.set_title("Validation IoU (mIoU)")
#     ax.legend(); ax.grid(alpha=0.3)
#     fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "iou_curve.png")); plt.close(fig)

#     # full breakdown -- val only now (train metrics aren't computed anymore, see train.py's main loop)
#     fig, ax = plt.subplots(figsize=(9, 5.5))
#     ax.plot(epochs, history["val_f1"], label="val F1")
#     ax.plot(epochs, history["val_iou"], label="val IoU")
#     ax.plot(epochs, history["val_precision"], label="val precision", alpha=0.5)
#     ax.plot(epochs, history["val_recall"], label="val recall", alpha=0.5)
#     ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1); ax.set_title("Validation metrics")
#     ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
#     fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "metrics_curve.png")); plt.close(fig)


# # ============================================================
# # TRAIN / EVAL LOOPS
# # ============================================================

# def _to_device(views: dict, device: str) -> dict:
#     return {name: (pre.to(device), post.to(device)) for name, (pre, post) in views.items()}


# def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
#     model.train()
#     total_loss = 0.0
#     for views, mask in tqdm(loader, desc="train", leave=False):
#         batch = _to_device(views, device)
#         mask = mask.to(device)

#         optimizer.zero_grad()
#         logits = model(batch)
#         loss = criterion(logits, mask)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * mask.size(0)
#     return total_loss / len(loader.dataset)


# @torch.no_grad()
# def evaluate_tiled(model, loader, device, model_names, criterion=None, threshold: float = 0.5):
#     model.eval()
#     all_probs, all_targets = [], []
#     total_loss, n_images = 0.0, 0

#     for pre_b, post_b, mask_b in tqdm(loader, desc="eval", leave=False):
#         pre_full, post_full, mask_full = pre_b[0], post_b[0], mask_b[0]
#         tiles, H, W, Hp, Wp, crop_px = tile_source(pre_full, post_full, model_names)

#         canvas = torch.zeros(Hp, Wp)
#         for views, top, left in tiles:
#             batch = {name: (p.unsqueeze(0).to(device), q.unsqueeze(0).to(device)) for name, (p, q) in views.items()}
#             logits = model(batch)[0, 0].cpu()
#             canvas[top:top + crop_px, left:left + crop_px] = logits

#         target_full = mask_full[:, :H, :W]
#         if criterion is not None:
#             loss = criterion(canvas[:H, :W].unsqueeze(0).unsqueeze(0), target_full.unsqueeze(0))
#             total_loss += loss.item()
#             n_images += 1

#         probs = torch.sigmoid(canvas[:H, :W]).numpy()
#         target = target_full[0].numpy()
#         all_probs.append(probs.ravel())
#         all_targets.append(target.ravel())

#     probs = np.concatenate(all_probs)
#     targets = np.concatenate(all_targets)
#     preds = (probs > threshold).astype(np.uint8)
#     val_loss = total_loss / n_images if n_images else None
#     return compute_metrics(preds, targets), probs, targets, val_loss


# @torch.no_grad()
# def evaluate_patches(model, loader, device, model_names, criterion=None, threshold: float = 0.5,
#                       ignore_index: float = -1.0):
#     model.eval()
#     all_probs, all_targets = [], []
#     total_loss, n_batches = 0.0, 0

#     for views, mask in tqdm(loader, desc="eval", leave=False):
#         batch = _to_device(views, device)
#         mask = mask.to(device)

#         logits = model(batch)
#         if criterion is not None:
#             loss = criterion(logits, mask)
#             total_loss += loss.item()
#             n_batches += 1

#         probs = torch.sigmoid(logits).cpu().numpy()
#         all_probs.append(probs.ravel())
#         all_targets.append(mask.cpu().numpy().ravel())

#     probs = np.concatenate(all_probs)
#     targets = np.concatenate(all_targets)
#     valid = targets != ignore_index
#     probs, targets = probs[valid], targets[valid]
#     preds = (probs > threshold).astype(np.uint8)
#     val_loss = total_loss / n_batches if n_batches else None
#     return compute_metrics(preds, targets), probs, targets, val_loss


# # ============================================================
# # MAIN
# # ============================================================

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--models", nargs="+", required=True, choices=list(MODEL_SPECS),
#                          help="one model for a baseline run, or several (from the SAME GSD group) to fuse")
#     parser.add_argument("--head", choices=HEAD_TYPES, default="linear_probe")
#     parser.add_argument("--fusion", choices=FUSION_STRATEGIES, default=None,
#                          help="required iff --models has more than one entry")
#     parser.add_argument("--epochs", type=int, default=40)
#     parser.add_argument("--batch_size", type=int, default=16)
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument("--weight_decay", type=float, default=1e-4)
#     parser.add_argument("--patience", type=int, default=8)
#     parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument("--croma_ckpt", default=os.path.join(
#         os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CROMA_large.pt"))
#     parser.add_argument("--unfreeze_encoders", action="store_true")
#     parser.add_argument("--proj_dim", type=int, default=256, help="early fusion: projected channel dim")
#     parser.add_argument("--dec_dim", type=int, default=128, help="sep_dec fusion: per-branch decoder channel dim")
#     parser.add_argument("--galileo_ckpt_dir", default=None,
#                          help="folder containing config.json + encoder.pt for a Galileo checkpoint size "
#                               "(nano/tiny/base/large), e.g. galileo/data/models/nano. Default: "
#                               "encoders.GalileoEncoder's built-in default path.")
#     parser.add_argument("--save_dir", default="models")
#     parser.add_argument("--patches_dir", default=None,
#                          help="directory of {split}_patches.pt from prepare_patches.py -- trains on ALL "
#                               "pre-tiled crops and batches val/test too, instead of one random crop per "
#                               "image per epoch + per-image tile/stitch eval")
#     parser.add_argument("--eval_batch_size", type=int, default=16)
#     parser.add_argument("--num_workers", type=int, default=8,
#                          help="DataLoader worker processes. 0 keeps data loading in the main "
#                               "process (safest but the GPU idles between batches); each worker "
#                               "gets its own reseeded random/numpy state (see loader._worker_init_fn) "
#                               "so augmentations stay independent across workers.")
#     args = parser.parse_args()

#     run_name = "+".join(args.models) + f"_{args.head}" + (f"_{args.fusion}" if args.fusion else "")
#     os.makedirs(args.save_dir, exist_ok=True)

#     if args.patches_dir:
#         train_loader, val_loader, test_loader = build_patch_loaders(
#             args.patches_dir, args.models, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
#             num_workers=args.num_workers,
#         )
#         evaluate = lambda *a, **kw: evaluate_patches(*a, model_names=args.models, **kw)
#         print(f"[{run_name}] pre-tiled patches from {args.patches_dir} "
#               f"({len(train_loader.dataset)} train / {len(val_loader.dataset)} val / "
#               f"{len(test_loader.dataset)} test patches)")
#     else:
#         train_loader, val_loader, test_loader = build_loaders(
#             args.models, batch_size=args.batch_size, num_workers=args.num_workers,
#         )
#         evaluate = lambda *a, **kw: evaluate_tiled(*a, model_names=args.models, **kw)

#     model = build_model(
#         args.models, args.head, args.fusion,
#         device=args.device, croma_ckpt=args.croma_ckpt, freeze_encoders=not args.unfreeze_encoders,
#         proj_dim=args.proj_dim, dec_dim=args.dec_dim,
#         galileo_checkpoint_folder=args.galileo_ckpt_dir,
#     )

#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     total = sum(p.numel() for p in model.parameters())
#     print(f"[{run_name}] ready | trainable params: {trainable:,} / {total:,}")

#     criterion = DiceBCELoss()
#     optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
#                                    lr=args.lr, weight_decay=args.weight_decay)

#     best_f1, best_epoch, patience_ct = 0.0, -1, 0
#     save_path = os.path.join(args.save_dir, f"best_model_{run_name}.pth")
#     plot_dir = os.path.join(args.save_dir, "plots", run_name)
#     history = {
#         "train_loss": [], "val_loss": [],
#         "val_f1": [], "val_f1_best_thr": [], "val_precision": [], "val_recall": [], "val_iou": [],
#         "threshold": [],
#     }

#     for epoch in range(args.epochs):
#         train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)

#         val_metrics, val_probs, val_targets, val_loss = evaluate(model, val_loader, args.device, criterion=criterion)
#         thr, thr_f1 = best_threshold(val_probs, val_targets.astype(np.uint8))

#         print(f"epoch {epoch + 1}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
#               f"val_f1(0.5)={val_metrics['f1']:.4f} | val_f1(best_thr={thr:.2f})={thr_f1:.4f} | "
#               f"val_iou={val_metrics['iou']:.4f}")

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["val_f1"].append(val_metrics["f1"])
#         history["val_f1_best_thr"].append(thr_f1)
#         history["val_precision"].append(val_metrics["precision"])
#         history["val_recall"].append(val_metrics["recall"])
#         history["val_iou"].append(val_metrics["iou"])
#         history["threshold"].append(thr)

#         plot_history(history, plot_dir)
#         with open(os.path.join(args.save_dir, f"history_{run_name}.json"), "w") as f:
#             json.dump(history, f, indent=2)

#         if thr_f1 > best_f1:
#             best_f1, best_epoch, patience_ct = thr_f1, epoch, 0
#             torch.save({"model": model.state_dict(), "threshold": thr, "epoch": epoch,
#                         "models": args.models, "head": args.head, "fusion": args.fusion}, save_path)
#         else:
#             patience_ct += 1
#             if patience_ct >= args.patience:
#                 print(f"Early stopping at epoch {epoch + 1} (best epoch {best_epoch + 1}, val_f1={best_f1:.4f})")
#                 break

#     ckpt = torch.load(save_path, map_location=args.device)
#     model.load_state_dict(ckpt["model"])
#     test_metrics, _, _, test_loss = evaluate(model, test_loader, args.device, criterion=criterion,
#                                               threshold=ckpt["threshold"])
#     print(f"\n{'=' * 60}\nTEST RESULTS | {run_name}\n{'=' * 60}")
#     print(f"  loss:      {test_loss:.4f}")
#     print(f"  F1:        {test_metrics['f1']:.4f}")
#     print(f"  mIoU:      {test_metrics['iou']:.4f}")
#     print(f"  precision: {test_metrics['precision']:.4f}")
#     print(f"  recall:    {test_metrics['recall']:.4f}")
#     print(f"  threshold: {ckpt['threshold']:.2f} (best on val)")

#     with open(os.path.join(args.save_dir, f"results_{run_name}.json"), "w") as f:
#         json.dump({"models": args.models, "head": args.head, "fusion": args.fusion,
#                     "best_epoch": best_epoch, "val_f1": best_f1,
#                     "test_metrics": test_metrics, "test_loss": test_loss, "threshold": ckpt["threshold"]},
#                    f, indent=2)


# if __name__ == "__main__":
#     main()


"""
fusion_v2/train.py
====================
Trains ANY single model or ANY fusion combination on CaBuAr wildfire
segmentation.

Usage:
    # single-model baselines
    python train.py --models croma --head linear_probe
    python train.py --models croma --head upernet
    python train.py --models prithvi --head upernet --unfreeze_encoders

    # fusion within the 10m group (croma/smarties/galileo) -- 30m group
    # (prithvi) currently has just one member, nothing to fuse there yet
    python train.py --models croma smarties galileo --head upernet --fusion early
    python train.py --models croma smarties galileo --head linear_probe --fusion sep_dec
    python train.py --models croma smarties galileo --head upernet --fusion late
"""

import os
import json
import argparse

import numpy as np
import torch
from tqdm import tqdm

from configs import MODEL_SPECS, HEAD_TYPES, FUSION_STRATEGIES
from loader import build_loaders, build_patch_loaders, tile_source
from models import build_model
from train_utils import DiceBCELoss, compute_metrics, best_threshold, plot_history


# ============================================================
# TRAIN / EVAL LOOPS
# ============================================================

def _to_device(views: dict, device: str) -> dict:
    return {name: (pre.to(device), post.to(device)) for name, (pre, post) in views.items()}


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
def evaluate_tiled(model, loader, device, model_names, criterion=None, threshold: float = 0.5):
    model.eval()
    all_probs, all_targets = [], []
    total_loss, n_images = 0.0, 0

    for pre_b, post_b, mask_b in tqdm(loader, desc="eval", leave=False):
        pre_full, post_full, mask_full = pre_b[0], post_b[0], mask_b[0]
        tiles, H, W, Hp, Wp, crop_px = tile_source(pre_full, post_full, model_names)

        canvas = torch.zeros(Hp, Wp)
        for views, top, left in tiles:
            batch = {name: (p.unsqueeze(0).to(device), q.unsqueeze(0).to(device)) for name, (p, q) in views.items()}
            logits = model(batch)[0, 0].cpu()
            canvas[top:top + crop_px, left:left + crop_px] = logits

        target_full = mask_full[:, :H, :W]
        if criterion is not None:
            loss = criterion(canvas[:H, :W].unsqueeze(0).unsqueeze(0), target_full.unsqueeze(0))
            total_loss += loss.item()
            n_images += 1

        probs = torch.sigmoid(canvas[:H, :W]).numpy()
        target = target_full[0].numpy()
        all_probs.append(probs.ravel())
        all_targets.append(target.ravel())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    preds = (probs > threshold).astype(np.uint8)
    val_loss = total_loss / n_images if n_images else None
    return compute_metrics(preds, targets), probs, targets, val_loss


@torch.no_grad()
def evaluate_patches(model, loader, device, model_names, criterion=None, threshold: float = 0.5,
                      ignore_index: float = -1.0):
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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODEL_SPECS),
                         help="one model for a baseline run, or several (from the SAME GSD group) to fuse")
    parser.add_argument("--head", choices=HEAD_TYPES, default="linear_probe")
    parser.add_argument("--fusion", choices=FUSION_STRATEGIES, default=None,
                         help="required iff --models has more than one entry")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--croma_ckpt", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CROMA_large.pt"))
    parser.add_argument("--unfreeze_encoders", action="store_true")
    parser.add_argument("--proj_dim", type=int, default=256, help="early fusion: projected channel dim")
    parser.add_argument("--dec_dim", type=int, default=128, help="sep_dec fusion: per-branch decoder channel dim")
    parser.add_argument("--galileo_ckpt_dir", default=None,
                         help="folder containing config.json + encoder.pt for a Galileo checkpoint size "
                              "(nano/tiny/base/large), e.g. galileo/data/models/nano. Default: "
                              "encoders.GalileoEncoder's built-in default path.")
    parser.add_argument("--save_dir", default="models")
    parser.add_argument("--patches_dir", default=None,
                         help="directory of {split}_patches.pt from prepare_patches.py -- trains on ALL "
                              "pre-tiled crops and batches val/test too, instead of one random crop per "
                              "image per epoch + per-image tile/stitch eval")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4,
                         help="DataLoader worker processes. 0 keeps data loading in the main "
                              "process (safest but the GPU idles between batches); each worker "
                              "gets its own reseeded random/numpy state (see loader._worker_init_fn) "
                              "so augmentations stay independent across workers.")
    args = parser.parse_args()

    run_name = "+".join(args.models) + f"_{args.head}" + (f"_{args.fusion}" if args.fusion else "")
    os.makedirs(args.save_dir, exist_ok=True)

    if args.patches_dir:
        train_loader, val_loader, test_loader = build_patch_loaders(
            args.patches_dir, args.models, batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
        )
        evaluate = lambda *a, **kw: evaluate_patches(*a, model_names=args.models, **kw)
        print(f"[{run_name}] pre-tiled patches from {args.patches_dir} "
              f"({len(train_loader.dataset)} train / {len(val_loader.dataset)} val / "
              f"{len(test_loader.dataset)} test patches)")
    else:
        train_loader, val_loader, test_loader = build_loaders(
            args.models, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        evaluate = lambda *a, **kw: evaluate_tiled(*a, model_names=args.models, **kw)

    model = build_model(
        args.models, args.head, args.fusion,
        device=args.device, croma_ckpt=args.croma_ckpt, freeze_encoders=not args.unfreeze_encoders,
        proj_dim=args.proj_dim, dec_dim=args.dec_dim,
        galileo_checkpoint_folder=args.galileo_ckpt_dir,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[{run_name}] ready | trainable params: {trainable:,} / {total:,}")

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=args.weight_decay)

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
                        "models": args.models, "head": args.head, "fusion": args.fusion}, save_path)
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
        json.dump({"models": args.models, "head": args.head, "fusion": args.fusion,
                    "best_epoch": best_epoch, "val_f1": best_f1,
                    "test_metrics": test_metrics, "test_loss": test_loss, "threshold": ckpt["threshold"]},
                   f, indent=2)


if __name__ == "__main__":
    main()