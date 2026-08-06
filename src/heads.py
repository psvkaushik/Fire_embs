"""
fusion_v2/heads.py
=====================
Two head styles, both taking a (B, C, g, g) token-grid feature map (from one
encoder, or an already-fused multi-encoder feature map) and producing
(B, 1, out_size, out_size) logits:

    LinearProbeHead  -- literally a linear probe: one 1x1 conv, C -> 1.
                        (a tiny 1-hidden-layer MLP variant is also offered
                        since a bare linear probe on raw ViT tokens is often
                        too weak to say much about representation quality.)

    UperNetHead      -- UperNet-*lite*. Real UperNet fuses features from
                        MULTIPLE backbone stages at different resolutions;
                        our frozen ViT encoders only expose a single final
                        token grid (see encoders.py), so this builds a
                        4-level pseudo-pyramid off that one grid via
                        strided up/downsampling (the standard SETR/ViT-Adapter
                        trick for running FPN-style heads on plain ViTs),
                        then does real PSP + top-down FPN fusion on top of
                        that pyramid. This is NOT the exact UperNet paper
                        architecture (which assumes a multi-stage CNN/Swin
                        backbone) -- flagging clearly so it's not mistaken
                        for a from-scratch multi-stage-feature UperNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbeHead(nn.Module):
    """out_size defaults to CROP_PX at call time via forward(..., out_size=)."""

    def __init__(self, in_dim: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(nn.Conv2d(in_dim, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1))
        else:
            self.net = nn.Conv2d(in_dim, 1, 1)   # true linear probe

    def forward(self, feat: torch.Tensor, out_size: int) -> torch.Tensor:
        logits = self.net(feat)
        return F.interpolate(logits, size=(out_size, out_size), mode="bilinear", align_corners=False)


class _PSPModule(nn.Module):
    """Pyramid Pooling Module (the "P" in UperNet)."""

    def __init__(self, in_dim: int, out_dim: int, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim), nn.GELU(),
            ) for s in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_dim + len(pool_scales) * out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [x] + [
            F.interpolate(stage(x), size=(h, w), mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(feats, dim=1))


class UperNetHead(nn.Module):
    def __init__(self, in_dim: int, fpn_dim: int = 256):
        super().__init__()
        # pseudo-pyramid: build 4 spatial scales off the single ViT grid
        self.to_4x = nn.Sequential(nn.ConvTranspose2d(in_dim, in_dim, 2, stride=2), nn.GELU(),
                                    nn.ConvTranspose2d(in_dim, in_dim, 2, stride=2))
        self.to_2x = nn.ConvTranspose2d(in_dim, in_dim, 2, stride=2)
        self.to_1x = nn.Identity()
        self.to_half = nn.MaxPool2d(2, 2)

        self.psp = _PSPModule(in_dim, fpn_dim)   # applied to the coarsest (0.5x) level

        # lateral 1x1s for the other 3 levels + top-down fusion convs
        self.lateral = nn.ModuleList([nn.Conv2d(in_dim, fpn_dim, 1) for _ in range(3)])
        self.fpn_fuse = nn.ModuleList([
            nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), nn.BatchNorm2d(fpn_dim), nn.GELU())
            for _ in range(3)
        ])

        self.fuse_all = nn.Sequential(
            nn.Conv2d(fpn_dim * 4, fpn_dim, 3, padding=1, bias=False), nn.BatchNorm2d(fpn_dim), nn.GELU(),
        )
        self.classifier = nn.Conv2d(fpn_dim, 1, 1)

    def forward(self, feat: torch.Tensor, out_size: int) -> torch.Tensor:
        p4x, p2x, p1x, phalf = self.to_4x(feat), self.to_2x(feat), self.to_1x(feat), self.to_half(feat)

        top = self.psp(phalf)   # coarsest level gets global context
        levels = []
        # top-down: coarse -> fine, upsample-add, in order [half, 1x, 2x, 4x]
        prev = top
        levels.append(prev)
        for raw, lat, fuse in zip([p1x, p2x, p4x], self.lateral, self.fpn_fuse):
            lat_feat = lat(raw)
            prev = F.interpolate(prev, size=lat_feat.shape[-2:], mode="bilinear", align_corners=False) + lat_feat
            prev = fuse(prev)
            levels.append(prev)

        target_hw = levels[-1].shape[-2:]   # finest level (4x) sets the common size
        merged = torch.cat(
            [F.interpolate(lv, size=target_hw, mode="bilinear", align_corners=False) for lv in levels], dim=1
        )
        logits = self.classifier(self.fuse_all(merged))
        return F.interpolate(logits, size=(out_size, out_size), mode="bilinear", align_corners=False)


def build_head(head_type: str, in_dim: int) -> nn.Module:
    if head_type == "linear_probe":
        return LinearProbeHead(in_dim)
    elif head_type == "upernet":
        return UperNetHead(in_dim)
    else:
        from configs import HEAD_TYPES
        raise ValueError(f"Unknown head_type '{head_type}' -- choose from {HEAD_TYPES}")