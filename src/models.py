"""
fusion_v2/models.py
======================
Factory: build_model(model_names, head_type, fusion_strategy, ...) -> one
nn.Module, whatever the combination. Resolves the run's shared footprint
(configs.footprint_m_for) ONCE, then builds every requested encoder at the
resize_px/grid_size that footprint implies for it:

    build_model(["croma"], "linear_probe")                             # standalone, footprint = croma's own native 1200m
    build_model(["croma","smarties","galileo"], "upernet", "early")    # 10m-group fusion, footprint = 2240m (smarties' native)
    build_model(["prithvi"], "upernet")                                 # 30m group currently has just one member
"""

import torch.nn as nn

from configs import MODEL_SPECS, HEAD_TYPES, FUSION_STRATEGIES, assert_fusable, footprint_m_for, native_crop_px_for
from encoders import build_encoder
from fusion import SingleModelWrapper, FUSION_CLASSES


def _build_encoders(model_names: list, footprint_m: float, device: str, croma_ckpt: str,
                     galileo_checkpoint_folder: str = None) -> dict:
    encoders = {}
    for name in model_names:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{name}' -- choose from {list(MODEL_SPECS)}")
        spec = MODEL_SPECS[name]
        if not spec.implemented:
            raise NotImplementedError(
                f"'{name}' isn't wired up yet -- see encoders.py's {name.capitalize()}Encoder "
                f"stub and configs.MODEL_SPECS['{name}'].notes for what's needed."
            )
        resize_px = spec.resize_px_for(footprint_m)
        encoders[name] = build_encoder(name, resize_px, device, croma_ckpt=croma_ckpt,
                                        galileo_checkpoint_folder=galileo_checkpoint_folder)
    return encoders


def build_model(
    model_names: list,
    head_type: str,
    fusion_strategy: str = None,
    device: str = "cuda",
    croma_ckpt: str = "CROMA_large.pt",
    freeze_encoders: bool = True,
    proj_dim: int = 256,
    dec_dim: int = 128,
    galileo_checkpoint_folder: str = None,
) -> nn.Module:
    if head_type not in HEAD_TYPES:
        raise ValueError(f"head_type '{head_type}' -- choose from {HEAD_TYPES}")

    footprint_m = footprint_m_for(model_names)   # raises a clear error if models don't share a group
    out_size = native_crop_px_for(footprint_m)   # mask/output resolution == native-storage crop size for this run

    # log how far each model is being stretched/shrunk from its own pretraining scale
    for name in model_names:
        spec = MODEL_SPECS[name]
        factor = spec.stretch_factor(footprint_m)
        tag = "native match" if abs(factor - 1.0) < 1e-6 else f"{factor:.2f}x native footprint"
        print(f"[build_model] {name}: footprint={footprint_m:.0f}m -> resize_px={spec.resize_px_for(footprint_m)} ({tag})")

    encoders = _build_encoders(model_names, footprint_m, device, croma_ckpt, galileo_checkpoint_folder)
    for enc in encoders.values():
        enc.set_frozen(freeze_encoders)

    if len(model_names) == 1:
        if fusion_strategy is not None:
            raise ValueError("fusion_strategy must be None for a single model -- there's nothing to fuse.")
        model = SingleModelWrapper(next(iter(encoders.values())), head_type, out_size)
    else:
        if fusion_strategy not in FUSION_STRATEGIES:
            raise ValueError(f"fusion_strategy '{fusion_strategy}' -- choose from {FUSION_STRATEGIES}")
        assert_fusable(model_names)
        cls = FUSION_CLASSES[fusion_strategy]
        kwargs = {"proj_dim": proj_dim} if fusion_strategy == "early" else {}
        kwargs = {"dec_dim": dec_dim} if fusion_strategy == "sep_dec" else kwargs
        model = cls(encoders, head_type, out_size, **kwargs)

    return model.to(device)