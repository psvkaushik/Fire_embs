"""
encoders.py
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
  - All three encoders (CROMA, SMARTIES, Copernicus-FM) are plain (non-temporal) 2D ViTs: batch pre+post
    together into one encoder call (2B images), split back apart,
    channel-concat -> embed_dim doubles.
"""

import os
import sys
from functools import partial

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
# CROMA
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
# SMARTIES
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
# Copernicus-FM (ViT-B/16)
#
# Wavelength/bandwidth-conditioned patch embed (DOFA-style hypernetwork), so
# all 12 CaBuAr S2 bands feed in directly. Code lives in the cloned repo
# <repo_root>/CopernicusFM_repo/Copernicus-FM/src (github.com/zhu-xlab/Copernicus-FM);
# weights from HF wangyi111/Copernicus-FM.
#
# The repo's model_vit.py uses relative imports (.dynamic_hypernetwork, ...) AND
# absolute ones (util.pos_embed, aurora.fourier, flexivit.*), and its folder is
# named `src` -- which clashes with this repo's own src/ -- so it's loaded under
# the package name `cfm_pkg` with its folder also appended to sys.path.
#
# return_intermediate=True gives the last block's patch tokens as a (B, C, h, w)
# grid (no cls token, no pooling). Pre/post batched like CROMA/SMARTIES.
# ============================================================

# Sentinel-2 central wavelength / bandwidth (nm), CABUAR_BANDS order (B01..B12, no B10)
_S2_WAVELENGTH_NM = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]
_S2_BANDWIDTH_NM = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 90, 180]


class CopernicusFMEncoder(_FrozenAwareModule):
    spec_name = "copernicusfm"

    def __init__(self, resize_px: int,
                 ckpt_path: str = os.path.join(_ROOT, "CopernicusFM_ViT_base_varlang_e100.pth"),
                 kernel_size: int = 16):
        super().__init__()
        import types
        spec = MODEL_SPECS["copernicusfm"]
        assert len(spec.bands) == len(_S2_WAVELENGTH_NM), "band list / wavelength table mismatch"
        src_dir = os.path.join(_ROOT, "CopernicusFM_repo", "Copernicus-FM", "src")
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(
                f"{src_dir} not found -- run: git clone https://github.com/zhu-xlab/Copernicus-FM.git "
                f"{os.path.join(_ROOT, 'CopernicusFM_repo')}")
        if src_dir not in sys.path:
            sys.path.append(src_dir)
        if "cfm_pkg" not in sys.modules:
            pkg = types.ModuleType("cfm_pkg")
            pkg.__path__ = [src_dir]
            sys.modules["cfm_pkg"] = pkg
        from cfm_pkg.model_vit import vit_base_patch16

        assert resize_px % kernel_size == 0, f"resize_px={resize_px} not divisible by kernel_size={kernel_size}"
        self.kernel_size = kernel_size
        self.patch_size = kernel_size
        self.grid_size = resize_px // kernel_size
        self.model = vit_base_patch16(img_size=resize_px, return_intermediate=True, intermediate_indices=[11])
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        # pos_embed is a fixed sin-cos table at the pretraining grid; the model resizes it at runtime,
        # so keep whatever the checkpoint holds and don't fail on a grid-size difference.
        sd = self.model.state_dict()
        for k in [k for k, v in state.items() if k in sd and v.shape != sd[k].shape]:
            print(f"CopernicusFM: dropping ckpt '{k}' {tuple(state[k].shape)} (model expects {tuple(sd[k].shape)})")
            del state[k]
        msg = self.model.load_state_dict(state, strict=False)
        print(f"CopernicusFM load_state_dict: missing={msg.missing_keys}, "
              f"unexpected={len(msg.unexpected_keys)} keys")

        self.register_buffer("_wvs", torch.tensor(_S2_WAVELENGTH_NM, dtype=torch.float32), persistent=False)
        self.embed_dim = self.model.blocks[0].norm1.normalized_shape[0] * 2   # pre+post concat
        self.frozen = False

    def _encode_one(self, imgs):
        meta = torch.full((imgs.shape[0], 4), float("nan"), device=imgs.device)   # lon/lat/time/area unknown
        _, feats = self.model(imgs, meta, _S2_WAVELENGTH_NM, _S2_BANDWIDTH_NM, None, "spectral", self.kernel_size)
        grid = feats[-1]
        # Block outputs are pre-final-norm and the ckpt ships no trained final norm (fc_norm.* missing),
        # so apply a plain per-token LayerNorm over channels.
        grid = torch.nn.functional.layer_norm(grid.permute(0, 2, 3, 1), (grid.shape[1],)).permute(0, 3, 1, 2)
        g = self.grid_size
        assert grid.shape[-2:] == (g, g), f"CopernicusFM: expected {g}x{g} grid, got {tuple(grid.shape[-2:])}"
        return grid

    def forward(self, pre, post):
        grid = self._encode_one(torch.cat([pre, post], dim=0))
        B = pre.shape[0]
        return torch.cat([grid[:B], grid[B:]], dim=1)


ENCODER_CLASSES = {
    "croma": CromaEncoder,
    "smarties": SmartiesEncoder,
    "copernicusfm": CopernicusFMEncoder,
}


def build_encoder(model_name: str, resize_px: int, device: str,
                   croma_ckpt: str = os.path.join(_ROOT, "CROMA_large.pt"),
                   copernicusfm_ckpt: str = os.path.join(_ROOT, "CopernicusFM_ViT_base_varlang_e100.pth")) -> nn.Module:
    cls = ENCODER_CLASSES[model_name]
    if model_name == "croma":
        enc = cls(resize_px, ckpt_path=croma_ckpt)
    elif model_name == "copernicusfm":
        enc = cls(resize_px, ckpt_path=copernicusfm_ckpt)
    else:
        enc = cls(resize_px)
    return enc.to(device)
