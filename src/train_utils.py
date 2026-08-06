"""
fusion_v2/train_utils.py
===========================
Loss, metrics, and plotting -- shared by both train.py (live encoders) and
train_cached.py (precomputed embeddings). Deliberately has NO import of
loader.py: loader.py's module-level code triggers an actual CaBuAr
download from HF Hub as an import side effect, and train_cached.py doesn't
need the raw dataset at all (only cached tensors + patch masks), so
anything train_cached.py imports needs to stay outside that chain.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# LOSS
# ============================================================

class DiceBCELoss(nn.Module):
    def __init__(self, ignore_index: float = -1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = (targets != self.ignore_index).float()
        targets = torch.where(valid.bool(), targets, torch.zeros_like(targets))

        bce_raw = self.bce(logits, targets)
        bce = (bce_raw * valid).sum() / valid.sum().clamp(min=1.0)

        probs = torch.sigmoid(logits).flatten(1) * valid.flatten(1)
        t = targets.flatten(1)
        inter = (probs * t).sum(dim=1)
        union = probs.sum(dim=1) + t.sum(dim=1)
        dice = 1 - (2 * inter + 1e-6) / (union + 1e-6)
        return 0.5 * bce + 0.5 * dice.mean()


# ============================================================
# METRICS
# ============================================================

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    preds = preds.astype(np.uint8).ravel()
    targets = targets.astype(np.uint8).ravel()
    return {
        "f1": f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall": recall_score(targets, preds, zero_division=0),
        "iou": jaccard_score(targets, preds, zero_division=0),
    }


def best_threshold(probs: np.ndarray, targets: np.ndarray):
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.1, 0.9, 17):
        f1 = f1_score(targets, (probs > t).astype(np.uint8), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


# ============================================================
# HISTORY / PLOTTING (val-only -- see train.py's history for why train
# metrics were dropped: they cost a full extra encoder pass per epoch for
# no real benefit over train_loss)
# ============================================================

def plot_history(history: dict, plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="train loss")
    ax.plot(epochs, history["val_loss"], label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training / validation loss")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "loss_curve.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_f1"], label="val F1 (thr=0.5)")
    ax.plot(epochs, history["val_f1_best_thr"], label="val F1 (best thr)", linestyle="--")
    ax.set_xlabel("epoch"); ax.set_ylabel("F1"); ax.set_ylim(0, 1); ax.set_title("Validation F1")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "f1_curve.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_iou"], label="val IoU")
    ax.set_xlabel("epoch"); ax.set_ylabel("IoU"); ax.set_ylim(0, 1); ax.set_title("Validation IoU (mIoU)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "iou_curve.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, history["val_f1"], label="val F1")
    ax.plot(epochs, history["val_iou"], label="val IoU")
    ax.plot(epochs, history["val_precision"], label="val precision", alpha=0.5)
    ax.plot(epochs, history["val_recall"], label="val recall", alpha=0.5)
    ax.set_xlabel("epoch"); ax.set_ylabel("score"); ax.set_ylim(0, 1); ax.set_title("Validation metrics")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "metrics_curve.png")); plt.close(fig)