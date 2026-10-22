"""
configs.py
======================
Single source of truth for:
  - CaBuAr's 12 Sentinel-2 channels: which S2 band each index is, its TRUE
    native GSD, and its center wavelength.
  - Per-model specs: bands consumed, target GSD, patch size, embed dim,
    NATIVE pretraining pixel count (native_px) -- used to pick group
    footprints and to report how far a given run stretches a model from
    what it actually saw in pretraining.
  - GSD groups + the physical footprint (in meters) each group crops from
    CaBuAr, chosen to fit the group's members as tightly as possible.

Nothing here does tensor work -- see resample.py / band_utils.py / loader.py.

Wired models: croma, smarties, copernicusfm (all in the 10m group).
"""

from dataclasses import dataclass
from typing import Optional

# ============================================================
# CROP GEOMETRY
# ============================================================

CROP_GSD_M = 20   # CaBuAr's stored per-pixel GSD (confirmed: 5490px tiles / 20m ~= 109.8km Sentinel-2 tile)

# ------------------------------------------------------------
# GROUP_FOOTPRINT_M -- the physical ground footprint (meters) cropped from
# CaBuAr for each GSD group. NOT an arbitrary shared number: derived from
# the group's own members' native pretraining resolution x native GSD.
#
# At CaBuAr's native CROP_GSD_M=20m/px storage grid, a given footprint_m
# implies crop_px = footprint_m / 20 (what's actually cropped out of the
# source image) and, per-model, resize_px = footprint_m / target_gsd_m
# (what's fed to that model, after resize_to() up/downsamples the crop).
# 10m targets need MORE pixels than the native 20m crop had (upsample). See ModelSpec.resize_px_for().
#
#   10m group -- croma / smarties / copernicusfm, anchored to SMARTIES' own native
#   footprint (2240m = 224px @ 10m):
#       smarties: 224px @ 10m (native)  -> EXACT match, stretch=1.0
#       croma:    120px @ 10m native -- stretched 2240/1200 = 1.87x beyond its
#                 own pretraining footprint (ALiBi's relative pos-encoding is
#                 what makes this a reasonable ask -- paper validates
#                 extrapolation well past this)
#     crop_px = 2240/20 = 112px pulled from CaBuAr, then resize_to() upsamples
#     that 112px crop to 224px (bilinear+antialias) for every 10m-group model.
#     112px is also the mask/output resolution (native_crop_px_for), since
#     out_size == crop_px throughout this codebase.
# ------------------------------------------------------------

GROUP_FOOTPRINT_M = {
    "10m": 2240,
}

# ============================================================
# CABUAR'S 12 CHANNELS
# ============================================================

@dataclass(frozen=True)
class BandInfo:
    index: int
    s2_band: str
    native_gsd_m: int
    wavelength_nm: int
    name: str


CABUAR_BANDS = [
    BandInfo(0,  "B01", 60, 443,  "aerosol"),
    BandInfo(1,  "B02", 10, 490,  "blue"),
    BandInfo(2,  "B03", 10, 560,  "green"),
    BandInfo(3,  "B04", 10, 665,  "red"),
    BandInfo(4,  "B05", 20, 705,  "red_edge_1"),
    BandInfo(5,  "B06", 20, 740,  "red_edge_2"),
    BandInfo(6,  "B07", 20, 783,  "red_edge_3"),
    BandInfo(7,  "B08", 10, 842,  "nir_broad"),
    BandInfo(8,  "B8A", 20, 865,  "nir_narrow"),
    BandInfo(9,  "B09", 60, 945,  "water_vapour"),
    BandInfo(10, "B11", 20, 1610, "swir_1"),
    BandInfo(11, "B12", 20, 2190, "swir_2"),
]

BAND_BY_S2ID = {b.s2_band: b for b in CABUAR_BANDS}
N_BANDS = len(CABUAR_BANDS)

RGB_S2_IDS = ["B04", "B03", "B02"]
ALL_12_S2_IDS = [b.s2_band for b in CABUAR_BANDS]

# ------------------------------------------------------------
# CaBuAr-wide approximate Sentinel-2 normalization stats (12 bands, CABUAR_BANDS
# order). Lives here (not just in loader.py) so encoders.py can reference it
# too, without importing loader.py itself -- loader.py's module-level code
# triggers an actual CaBuAr download from HF Hub as a side effect of import,
# which would be a bad thing to trigger just to read two constant tensors.
# ------------------------------------------------------------

CABUAR_S2_MEAN = [
    0.061, 0.090, 0.105, 0.122,
    0.164, 0.228, 0.250, 0.265,
    0.278, 0.289, 0.301, 0.315,
]
CABUAR_S2_STD = [
    0.043, 0.058, 0.067, 0.072,
    0.081, 0.091, 0.098, 0.103,
    0.109, 0.114, 0.119, 0.123,
]

# ============================================================
# PER-MODEL SPEC
#
# `native_px`  : the model's OWN pretraining pixel count at ITS native GSD
#                (informational, and the fallback footprint for standalone
#                runs of models that aren't in any FUSION_GROUPS entry).
# `resize_px_for(footprint_m)` / `grid_size_for(footprint_m)`: given a RUN's
#                chosen footprint (from GROUP_FOOTPRINT_M, or the model's own
#                native_px*target_gsd_m if run standalone unrouped), returns
#                the pixel size / token grid this model should actually see.
#                Rounded to the nearest multiple of patch_size so tokenization
#                is exact.
# ============================================================

@dataclass(frozen=True)
class ModelSpec:
    name: str
    bands: list
    target_gsd_m: float
    patch_size: int
    embed_dim: int
    native_px: int          # this model's own pretraining resolution
    weights: str
    notes: str = ""
    implemented: bool = False

    def resize_px_for(self, footprint_m: float) -> int:
        raw_px = footprint_m / self.target_gsd_m
        px = round(raw_px / self.patch_size) * self.patch_size
        return max(px, self.patch_size)

    def grid_size_for(self, footprint_m: float) -> int:
        return self.resize_px_for(footprint_m) // self.patch_size

    def native_footprint_m(self) -> float:
        return self.native_px * self.target_gsd_m

    def stretch_factor(self, footprint_m: float) -> float:
        """>1 means this run gives the model MORE ground context than it was
        pretrained on (extrapolation); <1 means less (safe for fixed sin-cos
        pos-embed models like smarties/copernicusfm, riskier for learned absolute
        pos-embed models unless they interpolate well)."""
        return footprint_m / self.native_footprint_m()


MODEL_SPECS = {
    # ---------------- 10m group ----------------
    "croma": ModelSpec(
        name="croma", bands=ALL_12_S2_IDS, target_gsd_m=10, patch_size=8, embed_dim=1024,
        native_px=120,
        weights="local checkpoint (CROMA_large.pt)",
        notes="ALiBi-based relative position encoding -- paper validates extrapolation up to 17.6x native size, "
              "and at this group's footprint (2240m vs croma's own native 1200m) it's stretched 1.87x, well "
              "within that validated range.",
        implemented=True,
    ),
    "smarties": ModelSpec(
        name="smarties", bands=ALL_12_S2_IDS, target_gsd_m=10, patch_size=16, embed_dim=1024,
        native_px=224,
        weights="gsumbul/SMARTIES-v1-ViT-L (HuggingFace, auto-download via transformers)",
        notes="NO explicit GSD input -- pretrained across BigEarthNet-MM S2 (10/20/60m), S1 (10m), and submeter "
              "RGB, so it has no single native footprint to be pulled away from on its own -- but its own "
              "native_px=224 @ 10m happens to equal 2240m, which is why the 10m group is anchored here rather "
              "than to croma. Fixed (non-learned) sin-cos pos-embed, recomputed exactly for whatever grid size "
              "results (see encoders.SmartiesEncoder) -- not an interpolation approximation the way a learned "
              "pos-embed would need.",
        implemented=True,
    ),
    "copernicusfm": ModelSpec(
        name="copernicusfm", bands=ALL_12_S2_IDS, target_gsd_m=10, patch_size=16, embed_dim=768,
        native_px=224,
        weights="wangyi111/Copernicus-FM (HF): CopernicusFM_ViT_base_varlang_e100.pth + "
                "github.com/zhu-xlab/Copernicus-FM cloned to <repo_root>/CopernicusFM_repo",
        notes="DOFA-style dynamic spectral patch embed (per-band central wavelength + bandwidth in nm), so all 12 "
              "CaBuAr S2 bands go in directly. Pretrained on Copernicus-Pretrain / SSL4EO-S12 S2 at 224px @ 10m "
              "(native footprint 2240m == this group's). Fixed sin-cos pos-embed, resized to the run's grid. "
              "ViT-B/16 -> 14x14 grid at 224px.",
        implemented=True,
    ),
}

MODEL_NAMES = list(MODEL_SPECS)

FUSION_GROUPS = {
    "10m": ["croma", "smarties", "copernicusfm"],
}

HEAD_TYPES = ("linear_probe", "upernet")
FUSION_STRATEGIES = ("early", "sep_dec", "late")


def group_of(model_name: str) -> Optional[str]:
    for g, names in FUSION_GROUPS.items():
        if model_name in names:
            return g
    return None


def assert_fusable(model_names: list) -> str:
    """Raises if model_names span more than one GSD group; returns the
    (single) group name otherwise."""
    groups = {group_of(m) for m in model_names}
    if None in groups or len(groups) > 1:
        raise ValueError(
            f"Models {model_names} -> groups {groups}. Fusion strategies in this codebase assume every "
            f"branch shares one target GSD/footprint (FUSION_GROUPS entries) -- fuse within a group."
        )
    return groups.pop()


def footprint_m_for(model_names: list) -> float:
    """The physical ground footprint (meters) this run's models should be
    cropped/resampled from. If every model shares a FUSION_GROUPS entry,
    uses that group's footprint. If it's a single model NOT in any group
    (currently none), falls back to that model's own native
    footprint so it can still be run standalone."""
    groups = {group_of(m) for m in model_names}
    if len(groups) == 1 and None not in groups:
        return GROUP_FOOTPRINT_M[groups.pop()]
    if len(model_names) == 1 and None in groups:
        return MODEL_SPECS[model_names[0]].native_footprint_m()
    raise ValueError(
        f"Can't determine a shared footprint for {model_names} -- either fuse models from the SAME "
        f"FUSION_GROUPS entry, or run a single model standalone."
    )


def native_crop_px_for(footprint_m: float) -> int:
    """The size (px, at CaBuAr's native 20m/px storage grid) of the crop to
    pull out of the source image for a run with this footprint."""
    px = footprint_m / CROP_GSD_M
    if px != int(px):
        raise ValueError(f"footprint_m={footprint_m} isn't a whole multiple of CROP_GSD_M={CROP_GSD_M}")
    return int(px)
