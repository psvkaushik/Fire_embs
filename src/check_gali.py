"""
Same embedding-variance check as diagnose_galileo.py, but parametrized by
--model_name so it can be run identically against croma/smarties (known
working, per your training curves) as a BASELINE, and against galileo, so
the two numbers are directly comparable -- a raw cosine similarity number
in isolation isn't diagnostic (ViT token embeddings are often high-cosine
across different inputs in general; what matters is whether galileo's
number is meaningfully HIGHER than croma/smarties' on the exact same crops).

Usage:
    python diagnose_embeddings.py --model_name croma
    python diagnose_embeddings.py --model_name smarties
    python diagnose_embeddings.py --model_name galileo
"""

import argparse
import torch

from encoders import build_encoder
from configs import MODEL_SPECS, footprint_m_for
from loader import build_loaders, build_patch_loaders


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True, choices=list(MODEL_SPECS))
    p.add_argument("--patches_dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--croma_ckpt", default="CROMA_large.pt")
    p.add_argument("--galileo_ckpt_dir", default=None)
    args = p.parse_args()
    name = args.model_name

    spec = MODEL_SPECS[name]
    footprint_m = footprint_m_for([name])
    resize_px = spec.resize_px_for(footprint_m)
    print(f"[{name}] footprint_m={footprint_m}  resize_px={resize_px}")

    kwargs = {}
    if name == "croma":
        kwargs["croma_ckpt"] ="/mnt/beegfs/spillal2/caburn/Emb_For_Fire/CROMA_large.pt"
    if name == "galileo":
        kwargs["galileo_checkpoint_folder"] = args.galileo_ckpt_dir
    enc = build_encoder(name, resize_px, args.device, **kwargs)
    enc.eval()

    if args.patches_dir:
        train_loader, _, _ = build_patch_loaders(args.patches_dir, [name], batch_size=4)
    else:
        train_loader, _, _ = build_loaders([name], batch_size=4)
    batches = [next(iter(train_loader)) for _ in range(2)]
    views_list = [b[0] for b in batches]

    feats = []
    with torch.no_grad():
        for views in views_list:
            pre, post = views[name]
            pre, post = pre.to(args.device), post.to(args.device)
            feat = enc(pre, post)
            feats.append(feat)
            print(f"[{name}] batch feat: shape={tuple(feat.shape)}  std={feat.std().item():.6f}")

    B = min(feats[0].shape[0], feats[1].shape[0])
    diff = feats[0][:B] - feats[1][:B]
    ratio = diff.abs().mean().item() / feats[0].std().item()
    print(f"[{name}] cross-batch |diff|/std ratio: {ratio:.4f}  (higher = embeddings vary more with input)")

    if feats[0].shape[0] > 1:
        v0 = feats[0][0, :, 0, 0]
        v1 = feats[0][1, :, 0, 0]
        cos = torch.nn.functional.cosine_similarity(v0, v1, dim=0)
        print(f"[{name}] within-batch cosine similarity (sample0 vs sample1, token 0,0): {cos.item():.4f}")


if __name__ == "__main__":
    main()