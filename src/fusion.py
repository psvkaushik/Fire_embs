"""
fusion_v2/fusion.py
======================
Two things live here:

  SingleModelWrapper -- run ANY one encoder standalone (linear_probe OR
                        upernet head), for the per-model baselines.

  Three fusion strategies over a GROUP of encoders that all share a
  target GSD (configs.assert_fusable enforces this -- see configs.py for
  why cross-GSD fusion isn't just silently allowed):

    EarlyFusion       -- project each branch to a common channel dim,
                         resize onto a common token grid, weighted-sum
                         (learnable softmax weights), ONE shared head.
                         (v1's "weighted_probe", generalized to N branches
                         and to either head type.)

    SepDecFusion      -- each branch gets its OWN head running independently
                         to (B, fpn_dim_or_1, out_size, out_size)-ish
                         intermediate features (not final logits), scaled by
                         a learnable softmax weight, channel-concatenated,
                         then a final small conv predicts the mask.
                         (v1's "weighted_decoder", generalized.)

    LateFusion        -- each branch gets its OWN full head all the way to
                         its OWN logits; final prediction is a learnable
                         softmax-weighted sum of per-branch logits. The
                         "purest" fusion in the sense that every branch
                         must independently be good at the task, not just
                         good in combination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import assert_fusable
from heads import build_head


# ============================================================
# SINGLE MODEL (per-model baseline)
#
# out_size has no default -- it's the mask/output resolution, which is now
# RUN-dependent (configs.native_crop_px_for(footprint_m_for(model_names))),
# not one global constant, since footprint differs by GSD group. Callers
# (models.py / train.py) must pass it explicitly.
# ============================================================

class SingleModelWrapper(nn.Module):
    def __init__(self, encoder: nn.Module, head_type: str, out_size: int):
        super().__init__()
        self.encoder = encoder
        self.head = build_head(head_type, encoder.embed_dim)
        self.out_size = out_size

    def forward(self, batch: dict) -> torch.Tensor:
        # batch: {model_name: (pre, post)} -- SingleModelWrapper only ever
        # reads its own one entry, keyed the same way fusion models are, so
        # train.py's batch-building code doesn't need to special-case it.
        (pre, post), = batch.values()
        feat = self.encoder(pre, post)
        return self.head(feat, self.out_size)


# ============================================================
# SHARED HELPERS
# ============================================================

def _common_grid(encoders: dict) -> int:
    return max(enc.grid_size for enc in encoders.values())


def _to_common(feat: torch.Tensor, common_grid: int) -> torch.Tensor:
    if feat.shape[-1] == common_grid:
        return feat
    return F.interpolate(feat, size=(common_grid, common_grid), mode="bilinear", align_corners=False)


class _SoftmaxWeights(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n))   # softmax -> equal weights at init

    def forward(self) -> torch.Tensor:
        return F.softmax(self.logits, dim=0)


# ============================================================
# EARLY FUSION
# ============================================================

class EarlyFusion(nn.Module):
    def __init__(self, encoders: dict, head_type: str, out_size: int, proj_dim: int = 256):
        super().__init__()
        assert_fusable(list(encoders.keys()))
        self.names = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.common_grid = _common_grid(encoders)
        self.out_size = out_size

        self.proj = nn.ModuleDict({
            name: nn.Conv2d(enc.embed_dim, proj_dim, 1) for name, enc in encoders.items()
        })
        self.weights = _SoftmaxWeights(len(self.names))
        self.head = build_head(head_type, proj_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        w = self.weights()
        fused = 0
        for i, name in enumerate(self.names):
            pre, post = batch[name]
            feat = _to_common(self.proj[name](self.encoders[name](pre, post)), self.common_grid)
            fused = fused + w[i] * feat
        return self.head(fused, self.out_size)


# ============================================================
# SEPARATE-DECODER FUSION
# ============================================================

class SepDecFusion(nn.Module):
    def __init__(self, encoders: dict, head_type: str, out_size: int, dec_dim: int = 128):
        super().__init__()
        assert_fusable(list(encoders.keys()))
        self.names = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.common_grid = _common_grid(encoders)
        self.out_size = out_size

        # per-branch head produces a dec_dim-channel feature map (not final
        # logits yet) at the branch's own grid; a linear_probe "head" here
        # just means a 1x1-conv projection, upernet means the full FPN body
        # with its classifier swapped for a dec_dim-channel projection.
        self.branch_heads = nn.ModuleDict({
            name: build_head(head_type, enc.embed_dim) for name, enc in encoders.items()
        })
        # branch_heads output 1 channel (segmentation logits) by default;
        # re-purpose as dec_dim-channel features via an extra projection so
        # concatenation carries more than a single scalar per branch.
        self.branch_proj = nn.ModuleDict({
            name: nn.Conv2d(1, dec_dim, 1) for name in self.names
        })
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
            pre, post = batch[name]
            branch_logits = self.branch_heads[name](self.encoders[name](pre, post), self.common_grid)
            feats.append(w[i] * self.branch_proj[name](branch_logits))
        fused = torch.cat(feats, dim=1)
        logits = self.merge(fused)
        return F.interpolate(logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)


# ============================================================
# LATE FUSION
# ============================================================

class LateFusion(nn.Module):
    def __init__(self, encoders: dict, head_type: str, out_size: int):
        super().__init__()
        assert_fusable(list(encoders.keys()))
        self.names = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.out_size = out_size
        self.branch_heads = nn.ModuleDict({
            name: build_head(head_type, enc.embed_dim) for name, enc in encoders.items()
        })
        self.weights = _SoftmaxWeights(len(self.names))

    def forward(self, batch: dict) -> torch.Tensor:
        w = self.weights()
        fused = 0
        for i, name in enumerate(self.names):
            pre, post = batch[name]
            logits = self.branch_heads[name](self.encoders[name](pre, post), self.out_size)
            fused = fused + w[i] * logits
        return fused


FUSION_CLASSES = {"early": EarlyFusion, "sep_dec": SepDecFusion, "late": LateFusion}