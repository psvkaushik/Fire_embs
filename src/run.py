"""
fusion_v2/run_10m_comparison.py
=================================
Runs the full "individually, then fused" comparison for the 10m group
(croma, smarties, galileo) and prints/saves a results table.

By default this just PRINTS the exact commands to run (so you can kick them
off yourself, in whatever order/parallelism suits your GPU setup -- these
are independent training runs, nothing here depends on run order). Pass
--execute to actually run them sequentially in-process instead.

Usage:
    # print the commands, run them yourself (recommended -- lets you
    # parallelize across GPUs, resume individual runs, etc.)
    python run_10m_comparison.py

    # or actually run everything sequentially, right now
    python run_10m_comparison.py --execute --epochs 40

    # once some/all runs have produced results_*.json in --save_dir:
    python run_10m_comparison.py --compare_only
"""

import argparse
import glob
import json
import os
import subprocess
import sys

INDIVIDUAL_MODELS = ["croma", "smarties", "galileo"]
HEADS = ["linear_probe", "upernet"]
FUSION_STRATEGIES = ["early", "sep_dec", "late"]

# Every fusion combo to run: pairs/triples that INCLUDE galileo (per request --
# not every possible subset, e.g. croma+smarties alone is deliberately excluded).
# Galileo is fed pre+post as one joint forward pass either way (native T=2 in
# GalileoEncoder, not two separate calls) -- this doesn't change standalone vs
# fused, so nothing special is needed here to get that; it's just how the
# encoder works whenever "galileo" appears in a models list at all.
FUSION_COMBOS = [
    ["croma", "galileo"],
    ["smarties", "galileo"],
    ["croma", "smarties"],
    ["croma", "smarties", "galileo"],
]


def build_configs(heads: list) -> list:
    """Returns list of (models, head, fusion) tuples: every individual model
    x every head (standalone, no fusion), then every FUSION_COMBOS entry x
    every fusion strategy x every head."""
    configs = []
    for model in INDIVIDUAL_MODELS:
        for head in heads:
            configs.append(([model], head, None))
    for combo in FUSION_COMBOS:
        for fusion in FUSION_STRATEGIES:
            for head in heads:
                configs.append((combo, head, fusion))
    return configs


def command_for(models: list, head: str, fusion: str, args) -> list:
    cmd = [sys.executable, "train.py", "--models", *models, "--head", head,
           "--epochs", str(args.epochs), "--save_dir", args.save_dir]
    if fusion:
        cmd += ["--fusion", fusion]
    if args.patches_dir:
        cmd += ["--patches_dir", args.patches_dir]
    return cmd


def run_name_for(models: list, head: str, fusion: str) -> str:
    name = "+".join(models) + f"_{head}" + (f"_{fusion}" if fusion else "")
    return name


def print_plan(configs: list, args):
    print(f"=== {len(configs)} runs in the 10m-group comparison ===\n")
    for models, head, fusion in configs:
        print(" ".join(command_for(models, head, fusion, args)))
    print(
        "\nThese are independent runs -- launch them in parallel across GPUs if you have more than one, "
        "or run sequentially. Once results_*.json files exist in --save_dir, run with --compare_only "
        "to see the aggregated table."
    )


def execute_plan(configs: list, args):
    for models, head, fusion in configs:
        name = run_name_for(models, head, fusion)
        print(f"\n{'=' * 60}\nRunning: {name}\n{'=' * 60}")
        cmd = command_for(models, head, fusion, args)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"WARNING: run '{name}' exited with code {result.returncode} -- continuing with the rest")


def compare(save_dir: str):
    paths = sorted(glob.glob(os.path.join(save_dir, "results_*.json")))
    if not paths:
        print(f"No results_*.json files found in {save_dir} yet.")
        return

    rows = []
    for path in paths:
        with open(path) as f:
            r = json.load(f)
        name = "+".join(r["models"]) + f"[{r['head']}]" + (f"[{r['fusion']}]" if r.get("fusion") else "")
        tm = r["test_metrics"]
        rows.append((name, tm["f1"], tm["precision"], tm["recall"], tm["iou"], r.get("test_loss")))

    rows.sort(key=lambda row: row[1], reverse=True)   # best F1 first

    header = f"{'config':40s} {'F1':>7s} {'precision':>10s} {'recall':>8s} {'IoU':>7s} {'loss':>8s}"
    print(header)
    print("-" * len(header))
    for name, f1, prec, rec, iou, loss in rows:
        loss_s = f"{loss:.4f}" if loss is not None else "n/a"
        print(f"{name:40s} {f1:7.4f} {prec:10.4f} {rec:8.4f} {iou:7.4f} {loss_s:>8s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--save_dir", default="models_10m_comparison")
    parser.add_argument("--patches_dir", default=None, help="e.g. patches/10m, from prepare_patches.py --group 10m")
    parser.add_argument("--heads", nargs="+", default=HEADS, choices=["linear_probe", "upernet"])
    parser.add_argument("--execute", action="store_true", help="actually run the configs, not just print them")
    parser.add_argument("--compare_only", action="store_true", help="skip running, just aggregate existing results")
    args = parser.parse_args()

    if args.compare_only:
        compare(args.save_dir)
        return

    configs = build_configs(args.heads)

    if args.execute:
        execute_plan(configs, args)
        print("\n\n=== FINAL COMPARISON ===")
        compare(args.save_dir)
    else:
        print_plan(configs, args)


if __name__ == "__main__":
    main()