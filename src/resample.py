"""
fusion_v2/resample.py
========================
Resamples a crop to an explicit target pixel size. Area-interpolation when
shrinking (genuinely averages pixels together, simulating a coarser
sensor), bilinear+antialias when enlarging (no real sub-pixel detail exists
to invent, just smooth upsampling).

Takes a plain size_px rather than a ModelSpec: which size a model gets
depends on the RUN's footprint (configs.footprint_m_for(model_names)), not
just the model in isolation -- see configs.py / loader.py for that.
"""

import torch
import torch.nn.functional as F


def resize_to(x: torch.Tensor, size_px: int) -> torch.Tensor:
    """x: (..., C, H, W) -> (..., C, size_px, size_px)."""
    if x.shape[-1] == size_px:
        return x
    orig_shape = x.shape
    x = x.reshape(-1, *orig_shape[-3:])
    mode = "area" if size_px < x.shape[-1] else "bilinear"
    kwargs = {"antialias": True} if mode == "bilinear" else {}
    x = F.interpolate(x, size=(size_px, size_px), mode=mode, **kwargs)
    return x.reshape(*orig_shape[:-2], size_px, size_px)