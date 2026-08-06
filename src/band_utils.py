"""
fusion_v2/band_utils.py
=========================
Turns a ModelSpec.bands list (S2 band ids, e.g. ["B02","B03","B04"]) into
the channel indices needed to slice CaBuAr's 12-band tensor, plus the
wavelength metadata that wavelength-conditioned encoders (Galileo) need.

This is the piece that replaces v1's hardcoded PRITHVI_BAND_INDICES with
something every model can share and that fails loudly (not silently) on a
typo'd band id.
"""

import torch

from configs import CABUAR_BANDS, BAND_BY_S2ID, ModelSpec

_S2ID_TO_CHANNEL = {b.s2_band: b.index for b in CABUAR_BANDS}


def band_indices(model_bands: list) -> list:
    """S2 band ids -> channel indices into CaBuAr's (12, H, W) tensor."""
    missing = [b for b in model_bands if b not in _S2ID_TO_CHANNEL]
    if missing:
        raise KeyError(
            f"Unknown S2 band id(s) {missing} -- valid ids are {list(_S2ID_TO_CHANNEL)}"
        )
    return [_S2ID_TO_CHANNEL[b] for b in model_bands]


def select_bands(x: torch.Tensor, model_bands: list) -> torch.Tensor:
    """x: (..., 12, H, W) -> (..., len(model_bands), H, W), reordered/subset
    to match model_bands exactly (order matters for models with fixed
    per-channel patch-embed weights, e.g. Prithvi)."""
    idx = band_indices(model_bands)
    return x[..., idx, :, :]


def wavelengths_nm(model_bands: list) -> list:
    """S2 band ids -> center wavelengths, for wavelength-conditioned
    encoders (e.g. Galileo) that need to know what each channel physically is."""
    return [BAND_BY_S2ID[b].wavelength_nm for b in model_bands]


def native_gsds_m(model_bands: list) -> list:
    """S2 band ids -> each band's TRUE native GSD (10/20/60m), independent
    of whatever target_gsd_m the model is being fed at. Useful for sanity
    checks / logging ("you're upsampling B11 4x to hit this model's GSD")."""
    return [BAND_BY_S2ID[b].native_gsd_m for b in model_bands]


def prepare_input(x: torch.Tensor, spec: ModelSpec) -> torch.Tensor:
    """Convenience: select this model's bands out of the full 12-channel
    tensor. Does NOT resample -- see resample.py for that; keeping band
    selection and GSD resampling as separate steps makes both easier to
    unit test independently."""
    return select_bands(x, spec.bands)