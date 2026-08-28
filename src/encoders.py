"""
fusion_v2/encoders.py
========================
One wrapper class per model, all sharing the same contract:

    encoder.embed_dim, encoder.patch_size, encoder.grid_size
    encoder(pre, post) -> (B, C, grid_size, grid_size) feature map

Pre/post are (B, len(spec.bands), resize_px, resize_px) -- already
band-selected (band_utils.select_bands) and already resampled to THIS RUN's
footprint (resample.resample_for_model), where resize_px/grid_size come
from configs.footprint_m_for(model_names) -- footprint depends on which
models are being run together (a group's shared footprint), not just the
model itself, so encoders take resize_px as a constructor arg instead of
reading a fixed value off ModelSpec.

Convention for combining pre/post:
  - Plain (non-temporal) 2D ViTs (CROMA, SatMAE, SMARTIES): batch pre+post
    together into one encoder call (2B images), split back apart,
    channel-concat -> embed_dim doubles.
  - Temporal-aware models (Prithvi, Galileo): stack pre/post as a genuine
    2-frame temporal axis, mean-pool over time after the encoder ->
    embed_dim stays single (not doubled).
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from configs import MODEL_SPECS


class _FrozenAwareModule(nn.Module):
    """Keeps a wrapped pretrained submodule in eval() (no dropout/droppath)
    whenever it's frozen, even while the parent fusion model is in
    train() mode for its own trainable head."""

    def set_frozen(self, frozen: bool):
        self.frozen = frozen
        for p in self.model.parameters():
            p.requires_grad = not frozen
        if frozen:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "frozen", False):
            self.model.eval()
        return self


# ============================================================
# CROMA -- fully wired
# ============================================================

class CromaEncoder(_FrozenAwareModule):
    spec_name = "croma"

    def __init__(self, resize_px: int, ckpt_path: str = "CROMA_large.pt"):
        super().__init__()
        spec = MODEL_SPECS["croma"]
        from use_croma import PretrainedCROMA
        assert resize_px % spec.patch_size == 0
        self.model = PretrainedCROMA(
            pretrained_path=ckpt_path, size="large", modality="optical", image_resolution=resize_px
        )
        self.embed_dim = self.model.encoder_dim * 2   # pre+post concat
        self.patch_size = spec.patch_size
        self.grid_size = resize_px // spec.patch_size
        self.frozen = False

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([pre, post], dim=0)
        tokens = self.model(optical_images=stacked)["optical_encodings"]
        B2, N, C = tokens.shape
        g = self.grid_size
        assert N == g * g, f"CROMA: expected {g*g} tokens, got {N}"
        grid = tokens.reshape(B2, g, g, C).permute(0, 3, 1, 2)
        B = pre.shape[0]
        return torch.cat([grid[:B], grid[B:]], dim=1)


# ============================================================
# Prithvi -- fully wired
# ============================================================

class PrithviEncoder(_FrozenAwareModule):
    spec_name = "prithvi"

    def __init__(self, resize_px: int, in_chans: int = 6):
        super().__init__()
        spec = MODEL_SPECS["prithvi"]
        from hf_prithvi import PrithviViT
        self.model = PrithviViT(
            num_frames=2, img_size=resize_px, patch_size=spec.patch_size, in_chans=in_chans, coords_encoding=None,
        )
        self.embed_dim = self.model.embed_dim
        self.patch_size = spec.patch_size
        self.grid_size = resize_px // spec.patch_size
        self.frozen = False

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        x = torch.stack([pre, post], dim=2)
        tokens = self.model.forward_features(x)[-1]
        tokens = tokens[:, 1:, :]
        B, N, C = tokens.shape
        T, g = 2, self.grid_size
        assert N == T * g * g, f"Prithvi: expected {T*g*g} tokens, got {N}"
        x = tokens.reshape(B, T, g, g, C).mean(dim=1)
        return x.permute(0, 3, 1, 2)


# ============================================================
# SMARTIES -- fully wired
#
# gsumbul/SMARTIES-v1-ViT-L's own forward() hardcodes its position embedding
# to whatever grid it was constructed with -- but that pos_embed is a FIXED
# sin-cos embedding (nn.Parameter(..., requires_grad=False), filled once via
# get_2d_sincos_pos_embed at construction, never learned), so recomputing
# that SAME closed-form embedding at a different grid size is EXACT, not an
# approximation the way bicubic-interpolating a LEARNED embedding would be.
# This reimplements forward_encoder's patch-embed + pos-embed + transformer-
# block pipeline using the loaded model's own submodules/weights, with the
# grid size read from the actual input instead of pinned at construction.
# (Ported unchanged from v1 -- this logic doesn't depend on footprint choice.)
# ============================================================

def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = True) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


# ------------------------------------------------------------
# CaBuAr S2 band id -> SMARTIES spectral-group key. SMARTIES pools bands
# from MANY sensors (Sentinel-1/2, Landsat8, generic RGB, ...) into shared
# numbered spectral-group buckets (blue_1/2/3, near_infrared_1/2/3, ...) --
# the numbering is NOT simple ascending-wavelength order across sensors, it's
# whatever their spectrum_specs.yaml assigned per sensor-band. Verified
# directly against that file (each key's own 'name' field states which S2
# band it is, e.g. green_2's name is literally "B03 (green)") rather than
# inferred from wavelength -- see find_smarties_mapping.py. All 12 of
# CaBuAr's bands have a home here (B09 maps to short_wave_infrared_1, NOT a
# "water_vapour" key -- that guess before this was verified was wrong).
# ------------------------------------------------------------

_S2ID_TO_SMARTIES_NAME = {
    "B01": "aerosol",
    "B02": "blue_1",
    "B03": "green_2",
    "B04": "red_2",
    "B05": "red_edge_1",
    "B06": "red_edge_2",
    "B07": "near_infrared_2",
    "B08": "near_infrared_1",
    "B8A": "near_infrared_3",
    "B09": "short_wave_infrared_1",
    "B11": "short_wave_infrared_3",
    "B12": "short_wave_infrared_4",
}


def smarties_encode(model: nn.Module, imgs: torch.Tensor, bands: list, grid_size: int) -> torch.Tensor:
    valid_keys = set(model.projection_conversion)
    missing = [name for name in bands if name not in valid_keys]
    if missing:
        raise KeyError(
            f"SMARTIES band name mismatch: {missing} not found in model.projection_conversion "
            f"-- these were silently dropped before (band_utils' own philosophy is to fail loudly on "
            f"this, not lose channels quietly). Valid keys on this checkpoint: {sorted(valid_keys)}. "
            f"Update the CaBuAr S2-id -> SMARTIES-name mapping in encoders.SmartiesEncoder to match."
        )
    proj_indices = np.array([model.projection_conversion[name] for name in bands])

    B = imgs.shape[0]
    proj_indices_t = torch.as_tensor(
        np.tile(proj_indices.reshape(1, 1, 1, -1), (B, grid_size, grid_size, 1)).astype(np.int32),
        device=imgs.device,
    )

    img_patches = model.tensor_patchify(imgs)
    _, nb_h, nb_w, nb_bands, _, _ = img_patches.shape
    assert nb_h == grid_size and nb_w == grid_size, \
        f"grid_size={grid_size} but tensor_patchify produced {nb_h}x{nb_w} -- check patch_size divides the input"

    img_spectrum_embeds = torch.zeros(
        (B, nb_h, nb_w, nb_bands, model.embed_dim), device=imgs.device, dtype=model.model_dtype
    )
    for projection_idx in torch.unbind(torch.unique(proj_indices_t)):
        mask = (proj_indices_t == projection_idx)
        img_spectrum_embeds[mask] = model.spectrum_projection(img_patches[mask], projection_idx)

    img_embeddings = model.projection_scaler * img_spectrum_embeds.mean(dim=3)
    img_embeddings = img_embeddings.reshape(B, nb_h * nb_w, model.embed_dim)

    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, img_embeddings), dim=1)

    pos_embed = _get_2d_sincos_pos_embed(model.embed_dim, grid_size, cls_token=True)
    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(imgs.device)
    x = x + pos_embed
    x = model.pos_drop(x)

    for blk in model.blocks:
        x = blk(x)
    return model.norm(x)


class SmartiesEncoder(_FrozenAwareModule):
    spec_name = "smarties"

    def __init__(self, resize_px: int):
        super().__init__()
        from transformers import AutoModel
        spec = MODEL_SPECS["smarties"]
        self.model = AutoModel.from_pretrained("gsumbul/SMARTIES-v1-ViT-L", trust_remote_code=True)
        self.embed_dim = self.model.config.embed_dim * 2   # pre+post concat
        self.patch_size = self.model.config.patch_size
        assert resize_px % self.patch_size == 0, \
            f"resize_px={resize_px} not divisible by SMARTIES' actual patch_size={self.model.config.patch_size} " \
            f"-- configs.py assumed patch_size={spec.patch_size}, update it if the loaded model disagrees."
        self.grid_size = resize_px // self.patch_size
        # spec.bands holds S2 ids (band_utils' universal contract, shared with every other
        # model spec) -- translate to SMARTIES' own spectral-group vocabulary here, since
        # that's what smarties_encode()/model.projection_conversion actually indexes by.
        # THIS IS NOT A GUESS: verified directly against SMARTIES' own spectrum_specs.yaml
        # (gsumbul/SMARTIES-v1-ViT-L, trust_remote_code cache), which explicitly labels each
        # cross-sensor spectral-group key with which Sentinel-2 band it corresponds to (e.g.
        # green_2: "B03 (green)", not green_1 -- SMARTIES' numbering is NOT simple wavelength
        # order across sensors, so don't try to re-derive this from wavelengths alone).
        missing = [b for b in spec.bands if b not in _S2ID_TO_SMARTIES_NAME]
        if missing:
            raise KeyError(
                f"No known SMARTIES spectral-group mapping for S2 band(s) {missing} -- "
                f"_S2ID_TO_SMARTIES_NAME needs updating (re-run find_smarties_mapping.py "
                f"against this checkpoint's spectrum_specs.yaml)."
            )
        self.bands = [_S2ID_TO_SMARTIES_NAME[b] for b in spec.bands]
        self.frozen = False

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([pre, post], dim=0)
        tokens = smarties_encode(self.model, stacked, self.bands, self.grid_size)
        tokens = tokens[:, 1:, :]
        B2, N, C = tokens.shape
        g = self.grid_size
        assert N == g * g, f"SMARTIES: expected {g*g} tokens, got {N}"
        grid = tokens.reshape(B2, g, g, C).permute(0, 3, 1, 2)
        B = pre.shape[0]
        return torch.cat([grid[:B], grid[B:]], dim=1)

# ============================================================
# Clay v1.5 -- fully wired
#
# Dynamic wavelength-conditioned embedding block -- passes central
# wavelength per band instead of fixed channel counts, so all 12
# CaBuAr S2 bands work directly, no spectral adapter needed.
# patch_size=8, embed_dim=1024 -- matches CROMA exactly, so token
# grids align for early fusion with no interpolation.
#
# IMPORTANT: Clay's encoder is an MAE encoder -- by default it masks
# 75% of patches and returns only the unmasked subset, which can't be
# reshaped into a full spatial grid. mask_ratio/shuffle are NOT forward()
# args -- they're set at LOAD time via load_from_checkpoint(), per
# Clay's own tutorials (see clay-foundation.github.io/model/tutorials/
# embeddings.html). mask_ratio=0.0 + shuffle=False gives back every
# patch, in raster order, so no unmasked_indices bookkeeping is needed.
#
# Pre/post handled the same batch-concat way as CROMA/SMARTIES.
# ============================================================

_CLAY_S2_WAVELENGTHS = [
    443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190,
]   # nanometres, CABUAR_BANDS order (B01..B12) -- see configs.CABUAR_BANDS


class ClayEncoder(_FrozenAwareModule):
    spec_name = "clay"

    def __init__(self, resize_px: int, ckpt_path: str = "clay-v1.5.ckpt",
                 metadata_path: str = "metadata.yaml"):
        super().__init__()
        spec = MODEL_SPECS["clay"]
        from claymodel.module import ClayMAEModule
        clay_module = ClayMAEModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            model_size="large",
            metadata_path=metadata_path,
            dolls=[16, 32, 64, 128, 256, 768, 1024],
            doll_weights=[1, 1, 1, 1, 1, 1, 1],
            mask_ratio=0.0,     # disable masking -- segmentation needs every patch
            shuffle=False,      # keep raster order -- no re-sorting via unmasked_indices
            map_location="cpu",
        )
        clay_module.eval()
        self.model = clay_module.model.encoder   # just the encoder, not the decoder

        assert resize_px % spec.patch_size == 0, \
            f"resize_px={resize_px} not divisible by Clay patch_size={spec.patch_size}"
        self.patch_size = spec.patch_size
        self.grid_size = resize_px // spec.patch_size
        self.embed_dim = spec.embed_dim * 2   # pre+post concat, same convention as CROMA/SMARTIES

        self.register_buffer(
            "_wavelengths",
            torch.tensor(_CLAY_S2_WAVELENGTHS, dtype=torch.float32)
        )
        self.frozen = False

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        stacked = torch.cat([pre, post], dim=0)   # (2B, 12, H, W)
        B2 = stacked.shape[0]

        waves = self._wavelengths.unsqueeze(0).expand(B2, -1).to(stacked.device)
        timestamps = torch.zeros(B2, 4, device=stacked.device)   # [week, hour, lat, lon] -- zeros ok

        tokens, unmsk_idx, msk_idx, msk_matrix = self.model(stacked, timestamps, waves)
        # with mask_ratio=0.0 + shuffle=False, tokens should be the FULL grid
        # in raster order -- unmsk_idx/msk_idx are ignorable here since
        # nothing was actually masked.

        B2_, N, C = tokens.shape
        g = self.grid_size
        if N == g * g + 1:
            tokens = tokens[:, 1:, :]   # strip CLS token if the encoder still prepends one
            B2_, N, C = tokens.shape
        assert N == g * g, \
            f"Clay: expected {g*g} tokens, got {N} -- check mask_ratio/shuffle took effect " \
            f"and resize_px={g*self.patch_size} is what was actually passed in"

        grid = tokens.reshape(B2_, g, g, C).permute(0, 3, 1, 2)
        B = pre.shape[0]
        return torch.cat([grid[:B], grid[B:]], dim=1)   # (B, 2048, g, g)
        
# ============================================================
# STUBS -- interface finalized, weight-loading left as TODO
# ============================================================

class _UnimplementedEncoder(_FrozenAwareModule):
    spec_name = "override_me"

    def __init__(self, resize_px: int):
        super().__init__()
        spec = MODEL_SPECS[self.spec_name]
        self.embed_dim = spec.embed_dim * 2   # assume non-temporal (pre+post concat); adjust if temporal
        self.patch_size = spec.patch_size
        assert resize_px % self.patch_size == 0
        self.grid_size = resize_px // self.patch_size
        self.resize_px = resize_px
        self.frozen = False
        self.model = nn.Identity()

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        spec = MODEL_SPECS[self.spec_name]
        raise NotImplementedError(
            f"'{self.spec_name}' encoder not wired yet. See configs.MODEL_SPECS['{self.spec_name}'].notes "
            f"for what's needed ({spec.weights}). Expected forward contract: pre/post are "
            f"(B, {len(spec.bands)}, {self.resize_px}, {self.resize_px}) -> return "
            f"(B, C, {self.grid_size}, {self.grid_size})."
        )


class SatMAEEncoder(_UnimplementedEncoder):
    spec_name = "satmae"
    # TODO: clone sustainlab-group/SatMAE, copy models_vit.py's multispectral ViT-Large, load the released
    # fMoW-Sentinel checkpoint, batch pre+post like CromaEncoder does.


# ============================================================
# Galileo -- best-effort real implementation, grounded directly in
# single_file_galileo.py's source (not just their docs/README, which
# describe a simplified [B,L,D] output that doesn't match what
# Encoder.forward() actually returns). NOT independently shape-verified
# against a live install -- see MODEL_SPECS['galileo'].notes.
#
# Genuinely temporal (H,W,T,C input), so pre/post become T=2 natively --
# no batch-concat trick needed the way croma/smarties use.
# Only the S2 band-groups are populated; Sentinel-1 and NDVI stay masked
# (zero-filled, unseen by the encoder) since CaBuAr has neither.
# ============================================================

# SPACE_TIME_BANDS / SPACE_TIME_BANDS_GROUPS_IDX layout, copied from
# single_file_galileo.py (S1_BANDS + S2_BANDS + ["NDVI"], 7 channel groups
# in this exact order) -- needed to build the masked input tensor without
# depending on their (non-batched) construct_galileo_input() helper.
_GALILEO_SPACE_TIME_BANDS = ["VV", "VH"] + [
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"
] + ["NDVI"]
_GALILEO_S2_GROUP_SLICE = slice(1, 6)   # S2_RGB, S2_Red_Edge, S2_NIR_10m, S2_NIR_20m, S2_SWIR (5 of 7 groups)
_GALILEO_S2_BAND_START = 2              # where our 10 S2 bands land inside the 13-band space-time vector

# ------------------------------------------------------------
# Galileo's OWN per-band normalization stats for its 10 S2 bands, sourced
# directly from their config/normalization.json (PRETRAINING_NORMALIZING_DICT
# in src/data/utils.py -- the "13" entry, S1(2)+S2(10)+NDVI(1), indices 2-11
# for our S2 bands, confirmed to be in the exact same order as
# configs.GALILEO_S2_BAND_IDS). Raw-DN-scale, NOT the ~0-1 reflectance scale
# loader.py's shared S2_MEAN/S2_STD produces -- see GalileoEncoder._renormalize.
# ------------------------------------------------------------


class GalileoEncoder(_FrozenAwareModule):
    spec_name = "galileo"

    def __init__(self, resize_px: int, checkpoint_folder: str = None):
        super().__init__()
        import sys as _sys
        from pathlib import Path
        _galileo_root = os.path.join(_ROOT, "galileo")   # expects the cloned repo at <repo_root>/galileo,
                                                            # i.e. a SIBLING of this src/ dir, not nested in it --
                                                            # _ROOT is derived from this file's own location so
                                                            # it's correct regardless of what this folder is named
        if _galileo_root not in _sys.path:
            _sys.path.insert(0, _galileo_root)
        from single_file_galileo import Encoder as GalileoRawEncoder

        if checkpoint_folder is None:
            checkpoint_folder = os.path.join(_galileo_root, "data", "models", "base")   # absolute, not
                                                                                          # cwd-dependent
        spec = MODEL_SPECS["galileo"]
        self.model = GalileoRawEncoder.load_from_folder(Path(checkpoint_folder), device=torch.device("cpu"))
        self.embed_dim = self.model.embedding_size   # read the real value -- checkpoint-size-dependent
        self.patch_size = spec.patch_size
        assert resize_px % self.patch_size == 0
        self.grid_size = resize_px // self.patch_size
        self.target_gsd_m = spec.target_gsd_m
        self.frozen = False

        # undo loader.py's generic CaBuAr-wide normalization (~0-1 reflectance scale), then reapply
        # Galileo's own raw-DN-scale stats (both live in configs.py as plain lists -- built into
        # tensors here, not there, per configs.py's own "no tensor work" convention).
        # loader_mean/std sliced+reordered from configs.CABUAR_S2_MEAN/STD (12-band CaBuAr order) to
        # this model's 10 S2 bands via band_utils.band_indices so the two normalization steps line up
        # band-for-band.
        from configs import CABUAR_S2_MEAN, CABUAR_S2_STD, GALILEO_S2_MEAN, GALILEO_S2_STD, GALILEO_DEFAULT_MONTH
        from band_utils import band_indices
        idx = band_indices(spec.bands)   # spec.bands is GALILEO_S2_BAND_IDS -- indices into the 12-band CaBuAr order
        loader_mean = torch.tensor(CABUAR_S2_MEAN)[idx].view(1, 10, 1, 1)
        loader_std = torch.tensor(CABUAR_S2_STD)[idx].view(1, 10, 1, 1)
        self.register_buffer("_loader_mean", loader_mean, persistent=False)
        self.register_buffer("_loader_std", loader_std, persistent=False)
        self.register_buffer("_galileo_mean", torch.tensor(GALILEO_S2_MEAN).view(1, 10, 1, 1), persistent=False)
        self.register_buffer("_galileo_std", torch.tensor(GALILEO_S2_STD).view(1, 10, 1, 1), persistent=False)
        self.default_month = GALILEO_DEFAULT_MONTH

    def _renormalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 10, H, W), already normalized by loader.py's shared CaBuAr-wide
        stats (which operate on CaBuAr's ~0-1 FRACTIONAL REFLECTANCE scale --
        that's why CABUAR_S2_MEAN's values are ~0.06-0.3, not thousands).
        Undoes that back to raw reflectance, then converts to the RAW DN scale
        Galileo's own stats assume (standard Sentinel-2 L2A convention: DN =
        reflectance * 10000 -- confirmed by GALILEO_S2_MEAN's blue-band value,
        ~1395, which is 0.1395 as a reflectance fraction, an entirely
        ordinary value) before applying Galileo's per-band normalization.
        Skipping the *10000 step would leave the recovered value off by ~4
        orders of magnitude from what Galileo's mean/std were computed for --
        the undo/redo algebra alone isn't enough, the unit conversion matters
        just as much."""
        raw_reflectance = x * self._loader_std + self._loader_mean
        raw_dn = raw_reflectance * 10000.0
        return (raw_dn - self._galileo_mean) / self._galileo_std

    def _build_masked_input(self, pre: torch.Tensor, post: torch.Tensor):
        """pre/post: (B, 10, H, W), our S2 band order (matches Galileo's own
        S2_BANDS order exactly), still normalized with loader.py's shared
        stats at this point. Returns the 9 tensors Encoder.forward() needs,
        with only the S2 channel-groups unmasked."""
        pre, post = self._renormalize(pre), self._renormalize(post)

        B, C, H, W = pre.shape
        device, dtype = pre.device, pre.dtype
        T = 2

        s_t_x = torch.zeros(B, H, W, T, len(_GALILEO_SPACE_TIME_BANDS), device=device, dtype=dtype)
        s_t_x[..., 0, _GALILEO_S2_BAND_START:_GALILEO_S2_BAND_START + C] = pre.permute(0, 2, 3, 1)
        s_t_x[..., 1, _GALILEO_S2_BAND_START:_GALILEO_S2_BAND_START + C] = post.permute(0, 2, 3, 1)

        s_t_m = torch.ones(B, H, W, T, 7, device=device, dtype=dtype)   # 7 space-time channel groups; 1 = masked
        s_t_m[..., _GALILEO_S2_GROUP_SLICE] = 0                       # unmask (make visible) the 5 S2 groups

        sp_x = torch.zeros(B, H, W, 0, device=device, dtype=dtype)     # no space-only bands (SRTM/DW/WC) available
        sp_m = torch.ones(B, H, W, 3, device=device, dtype=dtype)      # 3 space-only groups, all masked
        t_x = torch.zeros(B, T, 0, device=device, dtype=dtype)         # no time-only bands (ERA5/TC/VIIRS) available
        t_m = torch.ones(B, T, 3, device=device, dtype=dtype)          # 3 time-only groups, all masked
        st_x = torch.zeros(B, 0, device=device, dtype=dtype)           # no static bands (landscan/latlon/DW/WC) available
        st_m = torch.ones(B, 4, device=device, dtype=dtype)            # 4 static groups (LS, location, DW_static, WC_static), all masked
        months = torch.full((B, T), self.default_month, dtype=torch.long, device=device)   # their own
                                                                                              # DEFAULT_MONTH
                                                                                              # fallback (5),
                                                                                              # not 0

        return s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = self._build_masked_input(pre, post)

        s_t_out, *_ = self.model(
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months,
            patch_size=self.patch_size, input_resolution_m=self.target_gsd_m,
        )
        # s_t_out: (B, H', W', T, 7, D) -- mean-pool over the 5 active S2
        # groups, then over T (genuinely temporal, same convention as Prithvi)
        s2_groups = s_t_out[..., _GALILEO_S2_GROUP_SLICE, :]   # (B, H', W', T, 5, D)
        pooled = s2_groups.mean(dim=(3, 4))                     # (B, H', W', D) -- mean over T and groups together
        return pooled.permute(0, 3, 1, 2)                       # (B, D, H', W')


ENCODER_CLASSES = {
    "croma": CromaEncoder,
    "prithvi": PrithviEncoder,
    "smarties": SmartiesEncoder,
    "galileo": GalileoEncoder,
    "satmae": SatMAEEncoder,
    "clay": ClayEncoder,
}


def build_encoder(model_name: str, resize_px: int, device: str,
                   croma_ckpt: str = "/mnt/beegfs/spillal2/caburn/Emb_For_Fire/CROMA_large.pt",
                   galileo_checkpoint_folder: str = None,
                   clay_ckpt: str = "clay-v1.5.ckpt",
                   clay_metadata: str = "metadata.yaml") -> nn.Module:
    cls = ENCODER_CLASSES[model_name]
    if model_name == "croma":
        enc = cls(resize_px, ckpt_path=croma_ckpt)
    elif model_name == "galileo":
        kwargs = {"checkpoint_folder": galileo_checkpoint_folder} if galileo_checkpoint_folder else {}
        enc = cls(resize_px, **kwargs)
    elif model_name == "clay":
        enc = cls(resize_px, ckpt_path=clay_ckpt, metadata_path=clay_metadata)
    else:
        enc = cls(resize_px)
    return enc.to(device)
