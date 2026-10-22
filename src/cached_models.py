"""
cached_models.py
=============================
Model builders for training on PRECOMPUTED, frozen encoder embeddings (see
precompute_embeddings.py) instead of running encoders live every batch.

Same fusion mathematics as fusion.py's EarlyFusion/SepDecFusion/LateFusion
-- these classes reuse build_head/_SoftmaxWeights/_to_common directly from
there, not reimplemented -- just with the "run self.encoders[name](pre,
post)" step replaced by "read the precomputed feature map straight out of
the batch dict", since there's no live encoder here at all.

linear_probe ONLY, by design decision (this pipeline is deliberately
narrower in scope than fusion.py/models.py -- see the conversation this
came out of). build_head already supports "upernet" if that's ever wanted
here; nothing below hardcodes against it beyond always calling
build_head("linear_probe", ...).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import assert_fusable
from fusion import _SoftmaxWeights, _to_common
from heads import build_head

HEAD_TYPE = "linear_probe"


class CachedSingleModel(nn.Module):
    def __init__(self, embed_dim: int, out_size: int):
        super().__init__()
        self.head = build_head(HEAD_TYPE, embed_dim)
        self.out_size = out_size

    def forward(self, batch: dict) -> torch.Tensor:
        (feat,) = batch.values()
        return self.head(feat, self.out_size)


class CachedEarlyFusion(nn.Module):
    def __init__(self, embed_dims: dict, grid_sizes: dict, out_size: int, proj_dim: int = 256):
        super().__init__()
        assert_fusable(list(embed_dims.keys()))
        self.names = list(embed_dims.keys())
        self.out_size = out_size
        self.common_grid = max(grid_sizes.values())
        self.proj = nn.ModuleDict({name: nn.Conv2d(d, proj_dim, 1) for name, d in embed_dims.items()})
        self.weights = _SoftmaxWeights(len(self.names))
        self.head = build_head(HEAD_TYPE, proj_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        w = self.weights()
        fused = 0
        for i, name in enumerate(self.names):
            feat = _to_common(self.proj[name](batch[name]), self.common_grid)
            fused = fused + w[i] * feat
        return self.head(fused, self.out_size)


class CachedSepDecFusion(nn.Module):
    def __init__(self, embed_dims: dict, grid_sizes: dict, out_size: int, dec_dim: int = 128):
        super().__init__()
        assert_fusable(list(embed_dims.keys()))
        self.names = list(embed_dims.keys())
        self.out_size = out_size
        self.common_grid = max(grid_sizes.values())
        self.branch_heads = nn.ModuleDict({name: build_head(HEAD_TYPE, d) for name, d in embed_dims.items()})
        self.branch_proj = nn.ModuleDict({name: nn.Conv2d(1, dec_dim, 1) for name in self.names})
        self.weights = _SoftmaxWeights(len(self.names))
        self.merge = nn.Sequential(
            nn.Conv2d(dec_dim * len(self.names), dec_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dec_dim), nn.GELU(),
            nn.Conv2d(dec_dim, 1, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        w = self.weights()
        feats = []
        for i, name in enumerate(self.names):
            branch_logits = self.branch_heads[name](batch[name], self.common_grid)
            feats.append(w[i] * self.branch_proj[name](branch_logits))
        fused = torch.cat(feats, dim=1)
        logits = self.merge(fused)
        return F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


class CachedLateFusion(nn.Module):
    def __init__(self, embed_dims: dict, grid_sizes: dict, out_size: int):
        super().__init__()
        assert_fusable(list(embed_dims.keys()))
        self.names = list(embed_dims.keys())
        self.out_size = out_size
        self.branch_heads = nn.ModuleDict({name: build_head(HEAD_TYPE, d) for name, d in embed_dims.items()})
        self.weights = _SoftmaxWeights(len(self.names))

    def forward(self, batch: dict) -> torch.Tensor:
        w = self.weights()
        fused = 0
        for i, name in enumerate(self.names):
            logits = self.branch_heads[name](batch[name], self.out_size)
            fused = fused + w[i] * logits
        return fused


CACHED_FUSION_CLASSES = {"early": CachedEarlyFusion, "sep_dec": CachedSepDecFusion, "late": CachedLateFusion}


def build_cached_model(embed_dims: dict, grid_sizes: dict, out_size: int, fusion_strategy: str = None,
                        proj_dim: int = 256, dec_dim: int = 128, device: str = "cuda") -> nn.Module:
    model_names = list(embed_dims.keys())
    if len(model_names) == 1:
        if fusion_strategy is not None:
            raise ValueError("fusion_strategy must be None for a single model -- there's nothing to fuse.")
        model = CachedSingleModel(embed_dims[model_names[0]], out_size)
    else:
        if fusion_strategy not in CACHED_FUSION_CLASSES:
            raise ValueError(f"fusion_strategy '{fusion_strategy}' -- choose from {list(CACHED_FUSION_CLASSES)}")
        assert_fusable(model_names)
        cls = CACHED_FUSION_CLASSES[fusion_strategy]
        kwargs = {"proj_dim": proj_dim} if fusion_strategy == "early" else {}
        kwargs = {"dec_dim": dec_dim} if fusion_strategy == "sep_dec" else kwargs
        model = cls(embed_dims, grid_sizes, out_size, **kwargs)
    return model.to(device)