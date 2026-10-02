"""
run_croma_satmae.py
======================
End-to-end CROMA vs SatMAE vs CROMA+SatMAE fusion comparison on the 10m
group, using the existing cached-embedding pipeline:

  1. precompute_embeddings.precompute_one for croma and satmae (resume-aware:
     a model/split already cached in --embeddings_dir is skipped, so an
     existing croma cache is reused as-is).
  2. train_cached.py (linear probe) for:
       croma alone, satmae alone, croma+satmae x each --fusions strategy.
     Each run is its own subprocess so GPU memory is fully released between
     runs. A run whose results_{run_name}.json already exists in --save_dir
     is skipped unless --rerun is passed (so a cut-off job can just be
     resubmitted).
  3. Collects every results_*.json into one table, printed and written to
     {save_dir}/summary.json + summary.csv.

Usage:
    python run_croma_satmae.py --patches_dir patches/10m --embeddings_dir embeddings/10m \
        --croma_ckpt /path/to/CROMA_large.pt --satmae_ckpt /path/to/pretrain-vit-large-e199.pth \
        --save_dir results_croma_satmae --device cuda:0
"""

import argparse
import csv
import json
import os
import subprocess
import sys

import torch

from configs import FUSION_STRATEGIES
from precompute_embeddings import _default_croma_ckpt, precompute_one

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS = ["croma", "satmae"]


def _run_name(models: list, fusion: str = None) -> str:
    # must match train_cached.py's run_name exactly -- that's how results files are found
    return "+".join(models) + "_linear_probe" + (f"_{fusion}" if fusion else "")


def _train(models: list, fusion: str, args) -> str:
    run_name = _run_name(models, fusion)
    results_path = os.path.join(args.save_dir, f"results_{run_name}.json")
    if os.path.isfile(results_path) and not args.rerun:
        print(f"\n[{run_name}] results already at {results_path} -- skipping (pass --rerun to retrain)")
        return results_path

    cmd = [sys.executable, os.path.join(_SRC_DIR, "train_cached.py"),
           "--models", *models,
           "--embeddings_dir", args.embeddings_dir, "--patches_dir", args.patches_dir,
           "--epochs", str(args.epochs), "--batch_size", str(args.batch_size),
           "--eval_batch_size", str(args.eval_batch_size or args.batch_size),
           "--lr", str(args.lr), "--weight_decay", str(args.weight_decay),
           "--patience", str(args.patience), "--device", args.device,
           "--num_workers", str(args.num_workers), "--save_dir", args.save_dir]
    if fusion:
        cmd += ["--fusion", fusion]
    print(f"\n{'#' * 60}\n# {run_name}\n{'#' * 60}\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return results_path


def _summarize(results_paths: list, save_dir: str) -> None:
    rows = []
    for path in results_paths:
        with open(path) as f:
            r = json.load(f)
        m = r["test_metrics"]
        rows.append({
            "run": "+".join(r["models"]) + (f" ({r['fusion']})" if r.get("fusion") else ""),
            "test_f1": m["f1"], "test_iou": m["iou"], "test_precision": m["precision"],
            "test_recall": m["recall"], "val_f1": r["val_f1"], "threshold": r["threshold"],
            "best_epoch": r["best_epoch"] + 1,
        })

    header = f"{'run':<26}{'test F1':>9}{'mIoU':>9}{'prec':>9}{'recall':>9}{'val F1':>9}{'thr':>6}{'ep':>5}"
    print(f"\n{'=' * len(header)}\nCROMA / SatMAE / FUSION -- TEST RESULTS (linear probe)\n{'=' * len(header)}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['run']:<26}{row['test_f1']:>9.4f}{row['test_iou']:>9.4f}{row['test_precision']:>9.4f}"
              f"{row['test_recall']:>9.4f}{row['val_f1']:>9.4f}{row['threshold']:>6.2f}{row['best_epoch']:>5d}")
    best = max(rows, key=lambda r: r["test_f1"])
    print(f"\nbest by test F1: {best['run']} (F1={best['test_f1']:.4f}, mIoU={best['test_iou']:.4f})")

    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(save_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary written to {os.path.join(save_dir, 'summary.json')} / summary.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patches_dir", required=True, help="e.g. patches/10m, from prepare_patches.py --group 10m")
    p.add_argument("--embeddings_dir", required=True, help="e.g. embeddings/10m (existing croma cache is reused)")
    p.add_argument("--croma_ckpt", default=_default_croma_ckpt())
    p.add_argument("--satmae_ckpt", default="pretrain-vit-large-e199.pth")
    p.add_argument("--save_dir", default="results_croma_satmae")
    p.add_argument("--fusions", nargs="+", choices=FUSION_STRATEGIES, default=list(FUSION_STRATEGIES))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--embed_batch_size", type=int, default=8, help="batch size for the frozen-encoder pass")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--skip_precompute", action="store_true", help="embeddings already cached, go straight to training")
    p.add_argument("--rerun", action="store_true", help="retrain even if a results_*.json already exists")
    args = p.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not args.skip_precompute:
        os.makedirs(args.embeddings_dir, exist_ok=True)
        for model_name in MODELS:
            print(f"\n=== precompute: {model_name} ===")
            precompute_one(model_name, args.patches_dir, args.embeddings_dir, args.device,
                            args.embed_batch_size, args.croma_ckpt, galileo_ckpt_dir=None,
                            satmae_ckpt=args.satmae_ckpt)

    results_paths = [_train([m], None, args) for m in MODELS]
    results_paths += [_train(MODELS, fusion, args) for fusion in args.fusions]
    _summarize(results_paths, args.save_dir)


if __name__ == "__main__":
    main()
