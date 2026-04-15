#!/usr/bin/env python3
"""
model/swinunetr_backbone_utils.py

- Robust MONAI SwinUNETR construction across versions
- Safe swinViT feature extraction + layout conversion
- Safe loading of pretrained SwinUNETR weights:
    * shape-matched loading
    * patch-embed in_channel inflation (seg-pretrain in_ch=1 -> joint in_ch=2)
"""

from __future__ import annotations

import re
from typing import Tuple, Any, Dict, Optional, Set

import torch
import torch.nn as nn

try:
    from monai.networks.nets import SwinUNETR
except Exception as ex:
    raise RuntimeError("MONAI is required (monai.networks.nets.SwinUNETR).") from ex


def _drop_unexpected_kwargs(ctor, args, kwargs):
    while True:
        try:
            return ctor(*args, **kwargs)
        except TypeError as ex:
            msg = str(ex)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if m:
                bad = m.group(1)
                if bad in kwargs:
                    kwargs.pop(bad)
                    continue
            raise


def build_swinunetr_backbone(
    img_size: Tuple[int, int, int],
    in_channels: int,
    out_channels: int,
    feature_size: int,
    depths: Tuple[int, int, int, int],
    num_heads: Tuple[int, int, int, int],
    drop_rate: float,
    attn_drop_rate: float,
    dropout_path_rate: float,
    normalize: bool,
    use_checkpoint: bool,
    spatial_dims: int = 3,
) -> SwinUNETR:
    base_kwargs = dict(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        depths=depths,
        num_heads=num_heads,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        normalize=normalize,
        use_checkpoint=use_checkpoint,
    )

    # Variant A: no img_size kw
    try:
        return _drop_unexpected_kwargs(SwinUNETR, (), base_kwargs.copy())
    except TypeError:
        pass

    # Variant B: img_size kw exists
    kw2 = base_kwargs.copy()
    kw2["img_size"] = img_size
    try:
        return _drop_unexpected_kwargs(SwinUNETR, (), kw2)
    except TypeError:
        pass

    # Variant C: positional img_size,in_channels,out_channels
    kw3 = base_kwargs.copy()
    kw3.pop("in_channels", None)
    kw3.pop("out_channels", None)
    return _drop_unexpected_kwargs(SwinUNETR, (img_size, in_channels, out_channels), kw3)


def swinvit_features(backbone: SwinUNETR, x: torch.Tensor, normalize: bool):
    if not hasattr(backbone, "swinViT"):
        raise AttributeError("SwinUNETR has no attribute 'swinViT' (MONAI mismatch).")
    try:
        return backbone.swinViT(x, normalize)
    except TypeError:
        return backbone.swinViT(x)


def _expected_channels(feature_size: int, max_pow: int = 6) -> Set[int]:
    fs = int(feature_size)
    return {fs * (2 ** i) for i in range(max_pow + 1)}


def convert_swinvit_feats_to_channel_first(
    feats,
    expected_channels: Set[int],
    *,
    strict: bool = True,
    print_shapes: bool = False,
    tag: str = "swinViT",
):
    if not isinstance(feats, (list, tuple)):
        feats = [feats]

    out = []
    for i, f in enumerate(feats):
        if not torch.is_tensor(f):
            raise TypeError(f"[{tag}] feats[{i}] is not a torch.Tensor (got {type(f)}).")
        if f.ndim != 5:
            raise RuntimeError(f"[{tag}] feats[{i}] expected 5D, got shape={tuple(f.shape)}")

        if f.shape[1] in expected_channels and f.shape[-1] not in expected_channels:
            f2 = f
        elif f.shape[-1] in expected_channels and f.shape[1] not in expected_channels:
            f2 = f.permute(0, 4, 1, 2, 3).contiguous()
        elif f.shape[1] in expected_channels:
            f2 = f
        elif f.shape[-1] in expected_channels:
            f2 = f.permute(0, 4, 1, 2, 3).contiguous()
        else:
            msg = f"[{tag}] Cannot infer layout for feats[{i}] shape={tuple(f.shape)} expected_channels={sorted(expected_channels)}"
            if strict:
                raise RuntimeError(msg)
            f2 = f

        if print_shapes:
            print(f"[{tag}] feat[{i}] {tuple(f.shape)} -> {tuple(f2.shape)}")
        out.append(f2)
    return out


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state", "model", "state_dict", "net", "network"):
            v = ckpt.get(key, None)
            if isinstance(v, dict) and v and all(torch.is_tensor(x) for x in v.values()):
                return v
        if ckpt and all(torch.is_tensor(x) for x in ckpt.values()):
            return ckpt
    raise RuntimeError("Could not find a state_dict in checkpoint.")


def _find_patch_embed_weight_key(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    for k in sd.keys():
        if k.endswith("patch_embed.proj.weight"):
            return k
    return None


def _inflate_patch_embed_in_channels(cleaned_sd: Dict[str, torch.Tensor], model_sd: Dict[str, torch.Tensor], *, verbose: bool = True):
    k_ckpt = _find_patch_embed_weight_key(cleaned_sd)
    k_mod = _find_patch_embed_weight_key(model_sd)
    if k_ckpt is None or k_mod is None:
        return

    w_ckpt = cleaned_sd[k_ckpt]
    w_mod = model_sd[k_mod]
    if (not torch.is_tensor(w_ckpt)) or (not torch.is_tensor(w_mod)):
        return
    if w_ckpt.ndim != 5 or w_mod.ndim != 5:
        return

    in_ckpt = int(w_ckpt.shape[1])
    in_mod = int(w_mod.shape[1])
    if in_ckpt == in_mod:
        return

    # Safe, required rule: 1 -> 2 (CT only -> CT + union mask)
    if in_ckpt == 1 and in_mod > 1:
        w_new = torch.zeros_like(w_mod)
        w_new[:, :1].copy_(w_ckpt)
        w_new[:, 1:].zero_()
        cleaned_sd[k_mod] = w_new
        if k_ckpt != k_mod:
            cleaned_sd.pop(k_ckpt, None)
        if verbose:
            print(f"[SWIN][INFLATE] patch_embed in_ch {in_ckpt}->{in_mod} (copy ch0, zeros rest) key={k_mod}")
        return

    if verbose:
        print(f"[SWIN][INFLATE][SKIP] ckpt in_ch={in_ckpt}, model in_ch={in_mod} (no safe rule) key={k_mod}")


def load_swinunetr_pretrained(
    backbone: nn.Module,
    ckpt_path: str,
    *,
    verbose: bool = True,
    allow_inflate_patch_embed: bool = True,
) -> Dict[str, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_in = _extract_state_dict(ckpt)

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_in.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        if k2.startswith("backbone."):
            k2 = k2[len("backbone."):]
        cleaned[k2] = v

    model_sd = backbone.state_dict()
    if allow_inflate_patch_embed:
        _inflate_patch_embed_in_channels(cleaned, model_sd, verbose=verbose)

    filtered: Dict[str, torch.Tensor] = {}
    mismatched = 0
    not_in_model = 0
    for k, v in cleaned.items():
        if k not in model_sd:
            not_in_model += 1
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            mismatched += 1
            continue
        filtered[k] = v

    missing, unexpected = backbone.load_state_dict(filtered, strict=False)

    if verbose:
        print(
            f"[SWIN][LOAD] ckpt={ckpt_path} | in={len(cleaned)} matched={len(filtered)} "
            f"mismatched={mismatched} not_in_model={not_in_model} "
            f"missing_after={len(missing)} unexpected_after={len(unexpected)}"
        )

    return dict(
        ckpt_total=len(cleaned),
        matched=len(filtered),
        mismatched=mismatched,
        not_in_model=not_in_model,
        missing_after=len(missing),
        unexpected_after=len(unexpected),
    )
