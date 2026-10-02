"""
fusion_v2/configs.py
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

Currently wired: croma, smarties, prithvi (fully working), galileo (code
written and grounded in their actual source, but not yet shape-verified
against a live checkpoint -- see encoders.GalileoEncoder / SETUP.md).
satmae is wired into the 10m group (fusable with croma). scalemae/clay/dofa were removed
(see git history if you want them back) -- ScaleMAE was actually fully
wired via torchgeo, Clay/DOFA were stubs.
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
# 10m targets need MORE pixels than the native 20m crop had (upsample);
# 30m targets need FEWER (downsample). See ModelSpec.resize_px_for().
#
#   10m group -- croma / smarties / galileo, anchored to SMARTIES' own native
#   footprint (2240m = 224px @ 10m):
#       smarties: 224px @ 10m (native)  -> EXACT match, stretch=1.0
#       croma:    120px @ 10m native -- stretched 2240/1200 = 1.87x beyond its
#                 own pretraining footprint (ALiBi's relative pos-encoding is
#                 what makes this a reasonable ask -- paper validates
#                 extrapolation well past this)
#       galileo:  no fixed native image size (arbitrary H/W architecture) --
#                 BASE_GSD=10 is what matters (a literal constant in their
#                 source), and that's satisfied regardless of footprint
#     crop_px = 2240/20 = 112px pulled from CaBuAr, then resize_to() upsamples
#     that 112px crop to 224px (bilinear+antialias) for every 10m-group model.
#     112px is also the mask/output resolution (native_crop_px_for), since
#     out_size == crop_px throughout this codebase.
#
#   30m group -- prithvi, anchored to its own native footprint (6720m =
#   224px @ 30m, exact HLS pretraining config). Only one member currently,
#   so nothing to fuse within this group yet -- kept as a group (rather than
#   just a standalone model) so adding a second 30m-native model later is a
#   config change, not a refactor.
#     crop_px = 6720/20 = 336px pulled from CaBuAr, then resize_to() downsamples
#     (area interpolation) that 336px crop to 224px.
# ------------------------------------------------------------

GROUP_FOOTPRINT_M = {
    "10m": 2240,
    "30m": 6720,
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
GALILEO_S2_BAND_IDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]   # matches Galileo's S2_BANDS order exactly

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

# ------------------------------------------------------------
# Galileo's OWN per-band normalization stats, for the 10 S2 bands in
# GALILEO_S2_BAND_IDS order. Sourced directly from their
# config/normalization.json (PRETRAINING_NORMALIZING_DICT["13"], indices
# 2:12 -- the "13" entry covers space_time_x's 13 channels: VV, VH, then
# the 10 S2 bands in the same order as GALILEO_S2_BAND_IDS, then NDVI).
# Raw-reflectance-DN scale (means in the thousands) -- NOT compatible with
# CABUAR_S2_MEAN/STD's ~0-1 scale. encoders.GalileoEncoder undoes the
# CaBuAr-wide normalization and reapplies these before feeding Galileo.
# ------------------------------------------------------------

GALILEO_S2_MEAN = [
    1395.3408730676722, 1338.4026921784578, 1343.09883810357, 1543.8607982512297,
    2186.2022069512263, 2525.0932853316694, 2410.3377187373408, 2750.2854646886753,
    2234.911100061487, 1474.5311266077113,
]
GALILEO_S2_STD = [
    917.7041440370853, 913.2988423581528, 1092.678723527555, 1047.2206083460424,
    1048.0101611156767, 1143.6903026819996, 1098.979177731649, 1204.472755085893,
    1145.9774063078878, 980.2429840007796,
]
GALILEO_DEFAULT_MONTH = 5   # their own fallback (src/data/dataset.py DEFAULT_MONTH) when no real acquisition
                             # month is available -- NOT 0, which is what an earlier version of this code used

# ------------------------------------------------------------
# SatMAE's OWN per-band stats, for the 10 S2 bands in GALILEO_S2_BAND_IDS
# order (same order as MODEL_SPECS['satmae'].bands). Sourced directly from
# SatMAE/util/datasets.py's SentinelIndividualImageDataset.mean/.std (the
# 13-band fMoW-Sentinel stats, standard Sentinel-2 stack order B01, B02,
# ..., B8A, B09, B10, B11, B12), with indices [0, 9, 10] (B01/aerosol,
# B09/water_vapour, B10/cirrus) dropped -- exactly matching the official
# group_c pretraining command's `--dropped_bands 0 9 10`
# (SatMAE/README.md). Raw-DN-scale, NOT the ~0-1 reflectance scale
# loader.py's shared S2_MEAN/S2_STD produces.
#
# IMPORTANT: SatMAE does NOT z-score normalize -- its own SentinelNormalize
# transform (SatMAE/util/datasets.py) clips each band to
# [mean - 2*std, mean + 2*std] and affinely rescales that range to [0, 1]
# (originally to uint8 [0,255], then ToTensor's /255 -- composing those two
# steps is exactly a continuous rescale to [0,1], so the uint8 round trip
# can be skipped for inference). encoders.SatMAEEncoder._renormalize
# replicates this clip-and-rescale, NOT a z-score -- see GalileoEncoder's
# docstring for why reusing that pattern here would be wrong.
# ------------------------------------------------------------

SATMAE_S2_MEAN = [1184.3824625, 1120.77120066, 1136.26026392, 1263.73947144,
                  1645.40315151, 1846.87040806, 1762.59530783, 1972.62420416,
                  1732.16362238, 1247.91870117]
SATMAE_S2_STD = [650.2842772, 712.12507725, 965.23119807, 948.9819932,
                 1108.06650639, 1258.36394548, 1233.1492281, 1364.38688993,
                 1310.36996126, 1087.6020813]

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
        pos-embed models like smarties/prithvi, riskier for learned absolute
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
    "clay": ModelSpec(
        name="clay",
        bands=ALL_12_S2_IDS,        # Clay accepts all 12 bands via wavelength conditioning
        target_gsd_m=10,            # pretrained at 10m resolution on Sentinel-2
        patch_size=8,               # same as CROMA — produces 28x28 grid for 224px input
        embed_dim=1024,             # encoder dim from Clay v1.5 spec
        native_px=256,              # Clay was pretrained on 256x256 chips
        weights="made-with-clay/Clay v1.5 (HuggingFace, download clay-v1.5.ckpt)",
        notes="Dynamic wavelength-conditioned embedding block — accepts any band "
              "combination via central wavelength inputs, no spectral adapter needed. "
              "Dual MAE+DINOv2 pretraining objective. Patch size 8x8 matches CROMA "
              "exactly so token grids align for early fusion without interpolation.",
        implemented=True,
    ),
    "galileo": ModelSpec(
        name="galileo", bands=GALILEO_S2_BAND_IDS,
        target_gsd_m=10, patch_size=8, embed_dim=128,   # confirmed for the nano checkpoint via
                                                          # galileo_embed_size.py -- re-check if you switch to
                                                          # tiny/base/large, embed_dim is checkpoint-size-dependent
        native_px=64,   # NOT a hard architectural constraint -- Galileo has no fixed pretraining image size
                         # (arbitrary H/W, variable patch_size 4/8/16 per the paper's own compute/quality
                         # trade-off sweep). 64px is an informational reference point only (their EuroSAT eval
                         # benchmark chip size @ 10m = 640m), not a "native resolution" the way ViT models have
                         # one -- stretch_factor for this model is not a meaningful risk signal.
        weights="github.com/nasaharvest/galileo, single_file_galileo.py + a checkpoint folder "
                "(models/nano, models/tiny, ... from the HF repo nasaharvest/galileo/tree/main/models) -- "
                "Encoder.load_from_folder(folder, device)",
        notes="BASE_GSD=10 is a literal named constant in their source (not inferred from pixel-count math) -- "
              "passing target_gsd_m=10 as input_resolution_m is Galileo's actual documented default, zero "
              "extrapolation. Genuinely temporal architecture (H,W,T,C input) -- pre/post is fed as T=2 "
              "natively, NOT faked via batch-concat like croma/smarties. S2 bands only; Sentinel-1 and NDVI "
              "band-groups are left masked (zero-filled, ignored by the encoder) since CaBuAr has neither -- "
              "confirmed this degrades gracefully per their docs (<2% accuracy drop with half the modalities "
              "missing). Output is genuinely multi-group internally ([B,H',W',T,C_G,D] per "
              "src/single_file_galileo.py, traced through apply_linear_projection -> apply_attn -> "
              "split_and_expand_hwtc) -- encoders.GalileoEncoder mean-pools over the active S2 channel-groups "
              "and over T to get our usual (B,D,H',W'). NOT independently shape-verified against a live "
              "install -- the group-count and axis-order tracing is from reading their actual source, not "
              "guesswork, but hasn't been run. Get the real embedding_size for your checkpoint size with "
              "galileo_embed_size.py before trusting the placeholder embed_dim above.",
        implemented=True,
    ),

    "satmae": ModelSpec(
        name="satmae", bands=GALILEO_S2_BAND_IDS,   # group_c checkpoint: 13 S2 bands minus B01/B09/B10
        target_gsd_m=10, patch_size=8, embed_dim=1024, native_px=96,
        weights="sustainlab-group/SatMAE, models_vit_group_channels.py:vit_large_patch16 "
                "(group_c, --input_size 96 --patch_size 8), zenodo.org/record/7338613 pretrain-vit-large-e199.pth",
        notes="3 band-group patch embeds ((B02,B03,B04,B08), (B05,B06,B07,B8A), (B11,B12)), mean-pooled over "
              "groups after the transformer blocks (see encoders.SatMAEEncoder). Fixed 2D sin-cos pos-embed "
              "(requires_grad=False in pretraining), recomputed exactly for the 28x28 grid at this group's "
              "2240m footprint -- 2.33x its native 960m, same no-interpolation argument as smarties. "
              "Patch 8 @ 224px -> 28x28 grid, identical to croma's, so fusion needs no regridding.",
        implemented=True,
    ),

    # ---------------- 30m group ----------------
    "prithvi": ModelSpec(
        name="prithvi", bands=["B02", "B03", "B04", "B8A", "B11", "B12"], target_gsd_m=30, patch_size=16,
        embed_dim=1024, native_px=224,
        weights="local checkpoint, as in v1",
        notes="224px @ 30m is Prithvi's exact native HLS pretraining config -- exact match at 6720m, no "
              "extrapolation at all. Band order verify against your checkpoint: this uses B8A (narrow NIR) per "
              "the standard HLS-equivalent 6-band set; v1 used B08 (broad NIR) -- confirm which one your "
              "checkpoint actually trained on.",
        implemented=True,
    ),

    # ---------------- not currently in a fusion group ----------------
    "spectralgpt": ModelSpec(
        name="spectralgpt", bands=ALL_12_S2_IDS,
        target_gsd_m=10, patch_size=8, embed_dim=768, native_px=96,
        weights="danfenghong/IEEE_TPAMI_SpectralGPT, models_vit_tensor.py:vit_base_patch8, "
                "zenodo.org/records/8412455 SpectralGPT.pth",
        notes="3D spatial-spectral tubelet (8x8 spatial x 3-band spectral groups, 4 groups total), "
              "mean-pooled over the spectral-group axis after transformer blocks (see encoders.SpectralGptEncoder).",
        implemented=True,
    ),
}

MODEL_NAMES = list(MODEL_SPECS)

FUSION_GROUPS = {
    "10m": ["croma", "smarties", "galileo", "clay", "satmae"],
    "30m": ["prithvi"],
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
    (currently just spectralgpt), falls back to that model's own native
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
