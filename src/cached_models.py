"""
cached_models.py
================
Linear probes (and an optional temporal change head, see below) on PRECOMPUTED, frozen encoder embeddings (see
precompute_embeddings.py).

Every model here is: per-channel standardisation (non-trainable) -> ONE 1x1
conv over channels -> bilinear upsample of the logits. Nothing else is
trainable, so the trainable parameter count is exactly

    single:  C + 1
    fusion:  sum(C_i) + 1      (identical for early and late)

i.e. the only thing that differs between runs is which embeddings the probe
sees -- not how much extra capacity the fusion adds. (The old early/sep_dec
variants added projection convs, a 3x3 merge decoder, etc., which is why
param counts used to range from 5k to 1.4M and confounded the comparison.)

    early -- resample every embedding onto the largest token grid, concat on
             channels, one linear probe on the concatenation.
    late  -- one linear probe per embedding, logits upsampled to the output
             size and summed (one shared bias).

The two are the same function class (a linear map commutes with bilinear
resampling); they differ only in where the resampling happens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import assert_fusable
from fusion import _to_common


def _up(logits: torch.Tensor, out_size: int) -> torch.Tensor:
    return F.interpolate(logits, size=(out_size, out_size), mode="bilinear", align_corners=False)


class _Standardize(nn.Module):
    """Per-channel (x - mean) / std using running train-set statistics, no affine params."""

    def __init__(self, dim: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(dim, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class CachedSingleModel(nn.Module):
    def __init__(self, embed_dim: int, out_size: int):
        super().__init__()
        self.out_size = out_size
        self.norm = _Standardize(embed_dim)
        self.probe = nn.Conv2d(embed_dim, 1, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        (feat,) = batch.values()
        return _up(self.probe(self.norm(feat)), self.out_size)


class CachedEarlyFusion(nn.Module):
    def __init__(self, embed_dims: dict, grid_sizes: dict, out_size: int):
        super().__init__()
        assert_fusable(list(embed_dims.keys()))
        self.names = list(embed_dims.keys())
        self.out_size = out_size
        self.common_grid = max(grid_sizes.values())
        self.norms = nn.ModuleDict({n: _Standardize(d) for n, d in embed_dims.items()})
        self.probe = nn.Conv2d(sum(embed_dims.values()), 1, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        feats = [_to_common(self.norms[n](batch[n]), self.common_grid) for n in self.names]
        return _up(self.probe(torch.cat(feats, dim=1)), self.out_size)


class CachedLateFusion(nn.Module):
    def __init__(self, embed_dims: dict, grid_sizes: dict, out_size: int):
        super().__init__()
        assert_fusable(list(embed_dims.keys()))
        self.names = list(embed_dims.keys())
        self.out_size = out_size
        self.norms = nn.ModuleDict({n: _Standardize(d) for n, d in embed_dims.items()})
        self.probes = nn.ModuleDict({n: nn.Conv2d(d, 1, 1, bias=False) for n, d in embed_dims.items()})
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict) -> torch.Tensor:
        logits = sum(_up(self.probes[n](self.norms[n](batch[n])), self.out_size) for n in self.names)
        return logits + self.bias


# ============================================================
# TWO-LEVEL CHANGE MODEL
#
# Each cached embedding is [pre || post] on channels (first C/2 = pre, last C/2 = post).
#
# Level 1 -- temporal fusion INSIDE each model:
#     pre  -> pre_branch  (1x1 proj + depthwise 3x3) -> a
#     post -> post_branch (separate weights)         -> b
#     weighted concat: [w_pre * a , w_post * b]  (w = 2*softmax of 2 learnable scalars, init 1,1)
#   -> one change feature map (2*d channels) per model. Singles stop here (+ decoder).
#
# Level 2 -- fusion ACROSS models (pairs / triples), over the level-1 maps resampled to the largest grid:
#     early -- concat maps, one decoder
#     late  -- one decoder per model, logits averaged
#     attn  -- cross-attention: each model's tokens query the OTHER models' tokens (residual), then
#              concat + one decoder
#
# Hidden width d is the same for every model so head capacity doesn't depend on which encoder it is.
# ============================================================

class _TemporalBranch(nn.Module):
    def __init__(self, embed_dim: int, hidden: int):
        super().__init__()
        assert embed_dim % 2 == 0, "cached embedding must be [pre||post] on channels"
        c = embed_dim // 2
        self.norm = _Standardize(embed_dim)

        def branch():
            return nn.Sequential(
                nn.Conv2d(c, hidden, 1), nn.BatchNorm2d(hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden), nn.BatchNorm2d(hidden), nn.GELU(),
            )
        self.pre_branch, self.post_branch = branch(), branch()
        self.w_logits = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre, post = self.norm(x).chunk(2, dim=1)
        w = 2 * F.softmax(self.w_logits, dim=0)
        return torch.cat([w[0] * self.pre_branch(pre), w[1] * self.post_branch(post)], dim=1)


def _decoder(in_ch: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
                         nn.Conv2d(hidden, 1, 1))


class _CrossModelAttention(nn.Module):
    """Each model's tokens attend to the concatenated tokens of all OTHER models; residual + LayerNorm."""

    def __init__(self, names: list, dim: int, heads: int = 4):
        super().__init__()
        self.names = names
        self.attn = nn.ModuleDict({n: nn.MultiheadAttention(dim, heads, batch_first=True) for n in names})
        self.norm = nn.ModuleDict({n: nn.LayerNorm(dim) for n in names})

    def forward(self, feats: dict) -> dict:
        tokens = {n: f.flatten(2).transpose(1, 2) for n, f in feats.items()}   # (B, HW, d)
        out = {}
        for n in self.names:
            others = torch.cat([tokens[m] for m in self.names if m != n], dim=1)
            attended, _ = self.attn[n](tokens[n], others, others, need_weights=False)
            fused = self.norm[n](tokens[n] + attended)
            b, hw, d = fused.shape
            g = feats[n].shape[-1]
            out[n] = fused.transpose(1, 2).reshape(b, d, g, g)
        return out


class ChangeModel(nn.Module):
    def __init__(self, embed_dims: dict, grid_sizes: dict, out_size: int, fusion_strategy: str = None,
                 hidden: int = 128):
        super().__init__()
        if len(embed_dims) > 1:
            assert_fusable(list(embed_dims.keys()))
        self.names, self.out_size, self.fusion = list(embed_dims), out_size, fusion_strategy
        self.common_grid = max(grid_sizes.values())
        feat_dim = 2 * hidden
        self.branches = nn.ModuleDict({n: _TemporalBranch(d, hidden) for n, d in embed_dims.items()})
        if fusion_strategy == "late":
            self.decoders = nn.ModuleDict({n: _decoder(feat_dim, hidden) for n in self.names})
        else:   # single model, early, attn
            if fusion_strategy == "attn":
                self.cross = _CrossModelAttention(self.names, feat_dim)
            self.decoder = _decoder(feat_dim * len(self.names), hidden)

    def forward(self, batch: dict) -> torch.Tensor:
        feats = {n: _to_common(self.branches[n](batch[n]), self.common_grid) for n in self.names}
        if self.fusion == "late":
            return sum(_up(self.decoders[n](feats[n]), self.out_size) for n in self.names) / len(self.names)
        if self.fusion == "attn":
            feats = self.cross(feats)
        return _up(self.decoder(torch.cat([feats[n] for n in self.names], dim=1)), self.out_size)


CHANGE_FUSIONS = ("early", "late", "attn")


CACHED_FUSION_CLASSES = {"early": CachedEarlyFusion, "late": CachedLateFusion}


def build_cached_model(embed_dims: dict, grid_sizes: dict, out_size: int, fusion_strategy: str = None,
                        device: str = "cuda", head: str = "linear", hidden: int = 128) -> nn.Module:
    names = list(embed_dims.keys())
    if head == "change":
        if len(names) > 1 and fusion_strategy not in CHANGE_FUSIONS:
            raise ValueError(f"fusion_strategy '{fusion_strategy}' -- choose from {list(CHANGE_FUSIONS)}")
        if len(names) == 1 and fusion_strategy is not None:
            raise ValueError("fusion_strategy must be None for a single model -- there's nothing to fuse.")
        return ChangeModel(embed_dims, grid_sizes, out_size, fusion_strategy, hidden).to(device)
    if len(names) == 1:
        if fusion_strategy is not None:
            raise ValueError("fusion_strategy must be None for a single model -- there's nothing to fuse.")
        model = CachedSingleModel(embed_dims[names[0]], out_size)
    else:
        if fusion_strategy not in CACHED_FUSION_CLASSES:
            raise ValueError(f"fusion_strategy '{fusion_strategy}' -- choose from {list(CACHED_FUSION_CLASSES)}")
        model = CACHED_FUSION_CLASSES[fusion_strategy](embed_dims, grid_sizes, out_size)
    return model.to(device)
