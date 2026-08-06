"""
use_smarties.py
================
Adds multi-layer hidden-state access to a SMARTIES ViT
(gsumbul/SMARTIES-v1-ViT-L, loaded via
`AutoModel.from_pretrained(..., trust_remote_code=True)` in models.py).

SMARTIESHF.forward() (the vendored model code on the HF hub) only exposes
the pooled CLS feature or the final block's dense tokens (all_tokens=True) --
there's no hook for intermediate transformer-block outputs, unlike
PrithviViT.forward_features() or CROMA's forward_features() (added the same
way, in use_croma.py). Rather than patching the cached HF module file, this
re-implements SMARTIESHF.forward_encoder's patch-embedding + position-
embedding pipeline (same ops, running on the loaded model's own submodules/
parameters, so it uses the real pretrained weights) and returns EVERY
block's hidden state instead of only the last.
"""

import numpy as np
import torch


def smarties_forward_features(model, imgs: torch.Tensor, bands: list, is_patchify: bool = True) -> list:
    """
    Run `model` (a SMARTIESHF instance) up through all transformer blocks,
    returning the list of per-block hidden states -- (B, 1+num_patches, C)
    each, CLS token first, same token layout as
    SMARTIESHF.forward(..., all_tokens=True), but one entry per block
    instead of just the last.

    NOTE: none of these are passed through model.norm() (SMARTIESHF only
    LayerNorms the final block's output, inside forward_encoder's
    all_tokens branch) -- same convention as PrithviViT.forward_features(),
    LayerNorm each selected layer yourself downstream if needed (see
    SmartiesUPerNetSegProbe in models.py).
    """
    proj_indices = [
        model.projection_conversion[name]
        for name in bands if name in model.projection_conversion
    ]
    assert len(proj_indices) > 0, \
        "No valid bands provided -- check they match spectrum_specs.yaml band names."
    proj_indices = np.array(proj_indices)
    proj_indices = torch.as_tensor(
        np.tile(proj_indices.reshape(1, 1, 1, -1),
                (imgs.shape[0], model.nb_patch_length, model.nb_patch_length, 1)).astype(np.int32),
        device=imgs.device,
    )

    img_patches = model.tensor_patchify(imgs) if is_patchify else imgs
    B, nb_h, nb_w, nb_bands, _, _ = img_patches.shape
    device = img_patches.device

    img_spectrum_embeds = torch.zeros(
        (B, nb_h, nb_w, nb_bands, model.embed_dim), device=device, dtype=model.model_dtype
    )
    for projection_idx in torch.unbind(torch.unique(proj_indices)):
        mask = (proj_indices == projection_idx)
        img_spectrum_embeds[mask] = model.spectrum_projection(img_patches[mask], projection_idx)

    img_embeddings = model.projection_scaler * img_spectrum_embeds.mean(dim=3)
    img_embeddings = img_embeddings.reshape(-1, nb_h * nb_w, model.embed_dim)

    cls_tokens = model.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, img_embeddings), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    hidden_states = []
    for blk in model.blocks:
        x = blk(x)
        hidden_states.append(x)
    return hidden_states
