"""
preprocess.py
=============
Per-model input normalization for CaBuAr Sentinel-2 tensors.

The cached patches / loader tensors are `(DN - CABUAR_S2_MEAN) / CABUAR_S2_STD` -- a plain affine map of the raw
CaBuAr values, which are Sentinel-2 L2A digital numbers (reflectance * 10000; confirmed from the cached patches,
per-band means ~440-2000). Whatever units those two constants claim to be in, `x * STD + MEAN` recovers the DN
exactly, so every model below starts from DN and then applies the normalization its OWN pretraining used:

  croma        SSL4EO-S12 S2-L2A stats, per band mean +/- 2 std -> [0,1] (the SatMAE/SeCo recipe in CROMA's README)
  smarties     1st/99th-percentile min-max -> [0,1], then mean/std standardization (SMARTIES' S2-L2A pretraining
               spec, config/pretraining_sensors.yaml)
  copernicusfm z-score with Copernicus-Pretrain S2 stats (src/main_pretrain.py S2C_MEAN/STD)

All stats are for the 12 CaBuAr bands in CABUAR_BANDS order (B01..B12 without B10) unless noted.
"""

import torch
import torch.nn as nn

from configs import (CABUAR_S2_MEAN, CABUAR_S2_STD, CROMA_S2_MEAN, CROMA_S2_STD, SMARTIES_S2_MEAN, SMARTIES_S2_STD,
                     SMARTIES_S2_P1, SMARTIES_S2_P99, COPERNICUSFM_S2_MEAN, COPERNICUSFM_S2_STD)


def _t(v):
    return torch.tensor(v, dtype=torch.float32).view(1, -1, 1, 1)


class S2Normalizer(nn.Module):
    RECIPES = ("croma", "smarties", "copernicusfm")

    def __init__(self, recipe: str):
        super().__init__()
        assert recipe in self.RECIPES, recipe
        self.recipe = recipe
        self.register_buffer("_cb_mean", _t(CABUAR_S2_MEAN), persistent=False)
        self.register_buffer("_cb_std", _t(CABUAR_S2_STD), persistent=False)
        stats = {
            "croma": (CROMA_S2_MEAN, CROMA_S2_STD),
            "smarties": (SMARTIES_S2_MEAN, SMARTIES_S2_STD),
            "copernicusfm": (COPERNICUSFM_S2_MEAN, COPERNICUSFM_S2_STD),
        }[recipe]
        self.register_buffer("_m", _t(stats[0]), persistent=False)
        self.register_buffer("_s", _t(stats[1]), persistent=False)
        if recipe == "smarties":
            self.register_buffer("_p1", _t(SMARTIES_S2_P1), persistent=False)
            self.register_buffer("_p99", _t(SMARTIES_S2_P99), persistent=False)

    def to_dn(self, x: torch.Tensor) -> torch.Tensor:
        return x * self._cb_std + self._cb_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dn = self.to_dn(x)
        r = self.recipe
        if r == "croma":
            return ((dn - (self._m - 2 * self._s)) / (4 * self._s)).clamp(0, 1)
        if r == "smarties":
            u = ((dn - self._p1) / (self._p99 - self._p1)).clamp(0, 1)
            return (u - self._m) / self._s
        return (dn - self._m) / self._s                 # copernicusfm: plain z-score
