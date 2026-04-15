#!/usr/bin/env python3
"""Contour-aware SwinUNETR ROI token backbone.

Recommended end-to-end survival image path:
- CT-only shared SwinUNETR encoder
- low-resolution PT/LN localization heads on deep encoder features
- ROI tokenization from soft predicted PT/LN masks
- optional teacher forcing with GT PT/LN masks during training

Tokens (B,5,Dtok):
  0: GLOBAL
  1: PT_INTRA
  2: PT_PERI
  3: LN_INTRA
  4: LN_PERI
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from trifusesurv.models.swinunetr_backbone_utils import (
    _expected_channels,
    build_swinunetr_backbone,
    convert_swinvit_feats_to_channel_first,
    swinvit_features,
)
def masked_mean(feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = mask.clamp(0, 1)
    denom = w.sum(dim=(2, 3, 4)).clamp_min(eps)
    num = (feat * w).sum(dim=(2, 3, 4))
    return num / denom


def ct_stats_in_mask(ct: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    w = mask.clamp(0, 1)
    denom = w.sum(dim=(2, 3, 4)).clamp_min(eps)
    mu = (ct * w).sum(dim=(2, 3, 4)) / denom
    second = ((ct * ct) * w).sum(dim=(2, 3, 4)) / denom
    var = (second - mu * mu).clamp_min(0.0)
    sd = torch.sqrt(var + 1e-8)
    vol = w.mean(dim=(2, 3, 4))
    return torch.cat([mu, sd, vol], dim=1)


def ct_stats_global(ct: torch.Tensor, body: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    if body is None:
        mu = ct.mean(dim=(2, 3, 4))
        sd = ct.std(dim=(2, 3, 4)).clamp_min(1e-8)
        frac = ct.new_ones((ct.shape[0], 1))
        return torch.cat([mu, sd, frac], dim=1)

    w = body.clamp(0, 1)
    denom = w.sum(dim=(2, 3, 4)).clamp_min(eps)
    mu = (ct * w).sum(dim=(2, 3, 4)) / denom
    second = ((ct * ct) * w).sum(dim=(2, 3, 4)) / denom
    var = (second - mu * mu).clamp_min(0.0)
    sd = torch.sqrt(var + 1e-8)
    frac = w.mean(dim=(2, 3, 4))
    return torch.cat([mu, sd, frac], dim=1)


class AttnPool3D(nn.Module):
    def __init__(self, mask_bias: float = 2.0, temperature: float = 1.0):
        super().__init__()
        self.attn = nn.LazyConv3d(1, kernel_size=1, bias=False)
        self.mask_bias = float(mask_bias)
        self.temperature = float(temperature)

    def forward(self, feat: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.attn(feat)
        if mask is not None:
            logits = logits + self.mask_bias * mask.clamp(0, 1)
        w = torch.softmax((logits / self.temperature).flatten(2), dim=-1)
        feat_flat = feat.flatten(2)
        return (feat_flat * w).sum(dim=-1)


def interp_mask(mask: torch.Tensor, size: Tuple[int, int, int], mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.interpolate(mask, size=size, mode="nearest")
    if mode == "trilinear":
        return F.interpolate(mask, size=size, mode="trilinear", align_corners=False)
    raise ValueError(f"Unknown mask_interp mode: {mode}")


def binary_close(m: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return m
    k = 2 * r + 1
    d = (F.max_pool3d(m, kernel_size=k, stride=1, padding=r) > 0).float()
    e = 1.0 - (F.max_pool3d((1.0 - d).clamp(0, 1), kernel_size=k, stride=1, padding=r) > 0).float()
    return e


class ContourAwareROITokenBackbone(nn.Module):
    def __init__(
        self,
        *,
        img_size: Tuple[int, int, int],
        feature_size: int = 48,
        depths: Tuple[int, int, int, int] = (2, 2, 2, 2),
        num_heads: Tuple[int, int, int, int] = (3, 6, 12, 24),
        drop_rate: float = 0.10,
        attn_drop_rate: float = 0.10,
        dropout_path_rate: float = 0.20,
        normalize: bool = True,
        use_checkpoint: bool = False,
        token_dim: int = 512,
        token_mlp_dropout: float = 0.30,
        token_mlp_hidden_dim: int = 0,
        attn_mask_bias: float = 2.0,
        use_multiscale: bool = True,
        mask_interp: str = "nearest",
        min_roi_frac: float = 1e-5,
        min_roi_voxels_deep: int = 8,
        token_dropout: float = 0.05,
        pt_shell_radius: int = 3,
        ln_shell_radius: int = 3,
        shell_body_from_ct: bool = True,
        body_ct_thr: Union[str, float] = "auto",
        body_ct_thr_hu: float = -500.0,
        body_close_r: int = 2,
        body_max_frac: float = 0.995,
        strict_swinvit_layout: bool = True,
        debug_swinvit_layout: bool = False,
        force_presence_from_raw_masks: bool = False,
        raw_mask_threshold: float = 0.5,
        fallback_peri_to_intra: bool = True,
    ):
        super().__init__()
        self.normalize = bool(normalize)
        self.use_multiscale = bool(use_multiscale)
        self.mask_interp = str(mask_interp)
        self.min_roi_frac = float(min_roi_frac)
        self.min_roi_voxels_deep = int(max(min_roi_voxels_deep, 0))
        self.token_dropout = float(max(token_dropout, 0.0))
        self.pt_shell_radius = int(pt_shell_radius)
        self.ln_shell_radius = int(ln_shell_radius)
        self.shell_body_from_ct = bool(shell_body_from_ct)
        self.body_ct_thr = body_ct_thr
        self.body_ct_thr_hu = float(body_ct_thr_hu)
        self.body_close_r = int(body_close_r)
        self.body_max_frac = float(body_max_frac)
        self.strict_swinvit_layout = bool(strict_swinvit_layout)
        self.debug_swinvit_layout = bool(debug_swinvit_layout)
        self._checked_layout = False
        self._expected_c = _expected_channels(int(feature_size), max_pow=6)
        self.force_presence_from_raw_masks = bool(force_presence_from_raw_masks)
        self.raw_mask_threshold = float(raw_mask_threshold)
        self.fallback_peri_to_intra = bool(fallback_peri_to_intra)
        self._warned_sanitized = set()

        self.backbone_shared = build_swinunetr_backbone(
            img_size=tuple(img_size),
            in_channels=1,
            out_channels=2,
            feature_size=int(feature_size),
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            drop_rate=float(drop_rate),
            attn_drop_rate=float(attn_drop_rate),
            dropout_path_rate=float(dropout_path_rate),
            normalize=self.normalize,
            use_checkpoint=bool(use_checkpoint),
            spatial_dims=3,
        )

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.attn_pool = AttnPool3D(mask_bias=float(attn_mask_bias))
        self.loc_pt_head = nn.LazyConv3d(1, kernel_size=1, bias=True)
        self.loc_ln_head = nn.LazyConv3d(1, kernel_size=1, bias=True)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.GELU(),
            nn.Linear(64, 2),
        )
        self.max_tokens = 5
        self.token_dim = int(token_dim)

        token_mlp_hidden_dim = int(token_mlp_hidden_dim)
        if token_mlp_hidden_dim > 0:
            self.token_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.LazyLinear(token_mlp_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(token_mlp_dropout)),
                    nn.Linear(token_mlp_hidden_dim, self.token_dim),
                    nn.GELU(),
                    nn.Dropout(float(token_mlp_dropout)),
                    nn.LayerNorm(self.token_dim),
                ) for _ in range(self.max_tokens)
            ])
        else:
            self.token_mlp = nn.ModuleList([
                nn.Sequential(
                    nn.LazyLinear(self.token_dim),
                    nn.GELU(),
                    nn.Dropout(float(token_mlp_dropout)),
                    nn.LayerNorm(self.token_dim),
                ) for _ in range(self.max_tokens)
            ])

        self.token_type = nn.Parameter(torch.zeros(self.max_tokens, self.token_dim))
        nn.init.normal_(self.token_type, std=0.02)
    @property
    def out_dim(self) -> int:
        return int(self.token_dim)

    @property
    def num_tokens(self) -> int:
        return int(self.max_tokens)

    def iter_encoder_backbones(self):
        return [("backbone_shared", self.backbone_shared)]

    def _soft_shell(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        if int(radius) <= 0:
            return mask.new_zeros(mask.shape)
        k = 2 * int(radius) + 1
        dil = F.max_pool3d(mask.clamp(0, 1), kernel_size=k, stride=1, padding=int(radius))
        return (dil - mask).clamp(0, 1)

    def _ct_looks_hu(self, ct: torch.Tensor) -> bool:
        cmin = float(ct.amin().item())
        cmax = float(ct.amax().item())
        return (cmax > 50.0) or (cmin < -50.0)

    def _auto_body_thr(self, ct: torch.Tensor) -> float:
        return float(self.body_ct_thr_hu) if self._ct_looks_hu(ct) else 0.02

    def _deep_present(self, mask: torch.Tensor, deep_size: Tuple[int, int, int]) -> torch.Tensor:
        if self.min_roi_voxels_deep <= 0:
            return mask.new_ones((mask.shape[0],), dtype=torch.bool)
        m_ds = interp_mask(mask, size=deep_size, mode=self.mask_interp).clamp(0, 1)
        s = m_ds.sum(dim=(2, 3, 4)).squeeze(1)
        return s >= float(self.min_roi_voxels_deep)

    @staticmethod
    def _raw_present(mask01: torch.Tensor, thr: float) -> torch.Tensor:
        return (mask01 > float(thr)).flatten(1).any(dim=1)

    def _presence_from_mask(self, mask01: torch.Tensor, deep_size: Tuple[int, int, int]) -> torch.Tensor:
        mean_present = mask01.mean(dim=(2, 3, 4)).squeeze(1) > self.min_roi_frac
        return mean_present & self._deep_present(mask01, deep_size)

    def enable_mask_patch_embed_training(self, verbose: bool = True):
        if verbose:
            print("[PATCH][INFO] contour-aware CT-only encoder has no mask input channels; ignoring mask patch-embed training request.")

    def _sanitize_tensor(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        posinf: float = 0.0,
        neginf: float = 0.0,
        clamp_abs: float = 0.0,
    ) -> torch.Tensor:
        if not torch.isfinite(tensor).all().item():
            if name not in self._warned_sanitized:
                bad = int((~torch.isfinite(tensor)).sum().item())
                print(f"[WARN][CONTOUR] sanitized {bad} non-finite value(s) in {name}", flush=True)
                self._warned_sanitized.add(name)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=posinf, neginf=neginf)
        if float(clamp_abs) > 0.0:
            tensor = tensor.clamp(min=-float(clamp_abs), max=float(clamp_abs))
        return tensor

    def _sync_backbone_eval(self):
        self.backbone_shared.eval()

    def forward(
        self,
        x_img: torch.Tensor,
        *,
        mask_pt: Optional[torch.Tensor] = None,
        mask_ln: Optional[torch.Tensor] = None,
        teacher_force_alpha: float = 0.0,
        return_aux: bool = False,
    ):
        if x_img.ndim != 5 or x_img.size(1) != 1:
            raise ValueError(f"Expected contour-aware x_img (B,1,D,H,W), got {tuple(x_img.shape)}")

        B = x_img.size(0)
        ct = x_img[:, 0:1]

        if mask_pt is not None:
            mask_pt = mask_pt.to(device=ct.device, dtype=ct.dtype).clamp(0, 1)
        if mask_ln is not None:
            mask_ln = mask_ln.to(device=ct.device, dtype=ct.dtype).clamp(0, 1)

        body = None
        if self.shell_body_from_ct:
            thr = self._auto_body_thr(ct) if (isinstance(self.body_ct_thr, str) and self.body_ct_thr == "auto") else float(self.body_ct_thr)
            body = (ct > thr).float()
            if self.body_close_r > 0:
                body = binary_close(body, self.body_close_r)
            frac = body.mean(dim=(2, 3, 4)).squeeze(1)
            bad = (frac < 0.02)
            if 0.0 < self.body_max_frac < 1.0:
                bad = bad | (frac > self.body_max_frac)
            if bad.any():
                body = body.clone()
                body[bad] = 0.0
                if float(bad.float().mean().item()) > 0.5:
                    body = None

        feats = swinvit_features(self.backbone_shared, ct, self.normalize)
        feats = convert_swinvit_feats_to_channel_first(
            feats,
            self._expected_c,
            strict=self.strict_swinvit_layout,
            print_shapes=(self.debug_swinvit_layout and (not self._checked_layout)),
            tag="swinViT-SHARED",
        )
        feats = [self._sanitize_tensor(f, name=f"swin_feat_{i}") for i, f in enumerate(feats)]
        self._checked_layout = True

        use_feats = list(feats[-4:]) if (self.use_multiscale and len(feats) >= 4) else [feats[-1]]
        fdeep = use_feats[-1]
        deep_size = tuple(int(x) for x in fdeep.shape[2:])

        loc_pt_logits = self._sanitize_tensor(
            self.loc_pt_head(fdeep), name="loc_pt_logits", posinf=30.0, neginf=-30.0, clamp_abs=30.0
        )
        loc_ln_logits = self._sanitize_tensor(
            self.loc_ln_head(fdeep), name="loc_ln_logits", posinf=30.0, neginf=-30.0, clamp_abs=30.0
        )
        loc_pt_prob = torch.sigmoid(loc_pt_logits)
        loc_ln_prob = torch.sigmoid(loc_ln_logits)

        pt_prob = F.interpolate(loc_pt_prob, size=tuple(int(x) for x in ct.shape[2:]), mode="trilinear", align_corners=False).clamp(0, 1)
        ln_prob = F.interpolate(loc_ln_prob, size=tuple(int(x) for x in ct.shape[2:]), mode="trilinear", align_corners=False).clamp(0, 1)

        alpha = float(max(0.0, min(1.0, teacher_force_alpha)))
        pt_used = pt_prob
        ln_used = ln_prob
        if alpha > 0.0 and mask_pt is not None and mask_ln is not None:
            pt_used = (1.0 - alpha) * pt_prob + alpha * mask_pt
            ln_used = (1.0 - alpha) * ln_prob + alpha * mask_ln
        pt_used = pt_used.clamp(0, 1)
        ln_used = ln_used.clamp(0, 1)

        raw_pt_source = mask_pt if (mask_pt is not None and alpha > 0.0) else pt_used
        raw_ln_source = mask_ln if (mask_ln is not None and alpha > 0.0) else ln_used
        pt_present_raw = self._raw_present(raw_pt_source, self.raw_mask_threshold)
        ln_present_raw = self._raw_present(raw_ln_source, self.raw_mask_threshold)

        presence_logits = self._sanitize_tensor(
            self.presence_head(fdeep), name="presence_logits", posinf=30.0, neginf=-30.0, clamp_abs=30.0
        )
        pt_presence_logits = presence_logits[:, 0]
        ln_presence_logits = presence_logits[:, 1]
        pt_presence_pred = torch.sigmoid(pt_presence_logits) > 0.5
        ln_presence_pred = torch.sigmoid(ln_presence_logits) > 0.5

        pres_global = torch.ones(B, device=ct.device, dtype=torch.bool)
        pres_pt_intra = self._presence_from_mask(pt_used, deep_size) | pt_presence_pred
        pres_ln_intra = self._presence_from_mask(ln_used, deep_size) | ln_presence_pred

        pt_shell = self._soft_shell(pt_used, self.pt_shell_radius)
        ln_shell = self._soft_shell(ln_used, self.ln_shell_radius)
        if body is not None:
            pt_shell = pt_shell * body
            ln_shell = ln_shell * body

        if self.fallback_peri_to_intra:
            pt_shell_sum = pt_shell.sum(dim=(2, 3, 4)).squeeze(1)
            ln_shell_sum = ln_shell.sum(dim=(2, 3, 4)).squeeze(1)
            bad_pt_peri = pt_present_raw & (pt_shell_sum <= 0.0)
            bad_ln_peri = ln_present_raw & (ln_shell_sum <= 0.0)
            if bad_pt_peri.any():
                pt_shell = pt_shell.clone()
                pt_shell[bad_pt_peri] = pt_used[bad_pt_peri]
            if bad_ln_peri.any():
                ln_shell = ln_shell.clone()
                ln_shell[bad_ln_peri] = ln_used[bad_ln_peri]

        pres_pt_peri = self._presence_from_mask(pt_shell, deep_size) | pt_presence_pred
        pres_ln_peri = self._presence_from_mask(ln_shell, deep_size) | ln_presence_pred
        pres = torch.stack([pres_global, pres_pt_intra, pres_pt_peri, pres_ln_intra, pres_ln_peri], dim=1)

        if self.force_presence_from_raw_masks:
            pres = pres.clone()
            pres[:, 0] = True
            pres[:, 1] = pt_present_raw
            pres[:, 2] = pt_present_raw
            pres[:, 3] = ln_present_raw
            pres[:, 4] = ln_present_raw

        token_inputs: List[torch.Tensor] = []

        g_vecs = [self.gap(f).flatten(1) for f in use_feats]
        g = torch.cat(g_vecs, dim=1)
        g = torch.cat([g, ct_stats_global(ct, body=body)], dim=1)
        if body is not None:
            body_deep = interp_mask(body, size=fdeep.shape[2:], mode="nearest")
            g = torch.cat([g, self.attn_pool(fdeep, body_deep)], dim=1)
        else:
            g = torch.cat([g, self.attn_pool(fdeep, None)], dim=1)
        token_inputs.append(g)

        vecs = [ct_stats_in_mask(ct, pt_used)]
        for f in use_feats:
            vecs.append(masked_mean(f, interp_mask(pt_used, size=f.shape[2:], mode=self.mask_interp)))
        token_inputs.append(torch.cat(vecs + [self.attn_pool(fdeep, interp_mask(pt_used, size=fdeep.shape[2:], mode=self.mask_interp))], dim=1))

        vecs = [ct_stats_in_mask(ct, pt_shell)]
        for f in use_feats:
            vecs.append(masked_mean(f, interp_mask(pt_shell, size=f.shape[2:], mode=self.mask_interp)))
        token_inputs.append(torch.cat(vecs + [self.attn_pool(fdeep, interp_mask(pt_shell, size=fdeep.shape[2:], mode=self.mask_interp))], dim=1))

        vecs = [ct_stats_in_mask(ct, ln_used)]
        for f in use_feats:
            vecs.append(masked_mean(f, interp_mask(ln_used, size=f.shape[2:], mode=self.mask_interp)))
        token_inputs.append(torch.cat(vecs + [self.attn_pool(fdeep, interp_mask(ln_used, size=fdeep.shape[2:], mode=self.mask_interp))], dim=1))

        vecs = [ct_stats_in_mask(ct, ln_shell)]
        for f in use_feats:
            vecs.append(masked_mean(f, interp_mask(ln_shell, size=f.shape[2:], mode=self.mask_interp)))
        token_inputs.append(torch.cat(vecs + [self.attn_pool(fdeep, interp_mask(ln_shell, size=fdeep.shape[2:], mode=self.mask_interp))], dim=1))

        hs: List[torch.Tensor] = []
        for i in range(self.max_tokens):
            h = self.token_mlp[i](token_inputs[i]) + self.token_type[i].unsqueeze(0)
            if i > 0:
                absent = ~pres[:, i]
                if absent.any():
                    h = h.masked_fill(absent.unsqueeze(1), 0.0)
            hs.append(h)

        tok_img = torch.stack(hs, dim=1)
        tok_img = torch.nan_to_num(tok_img, nan=0.0, posinf=0.0, neginf=0.0)

        if self.training and self.token_dropout > 0:
            pres2m = pres.clone()
            tok2m = tok_img
            for tok_i in (1, 2, 3, 4):
                if pres2m[:, tok_i].any():
                    drop = (torch.rand(B, device=x_img.device) < self.token_dropout) & pres2m[:, tok_i]
                    if drop.any():
                        pres2m[drop, tok_i] = False
                        tok2m = tok2m.clone()
                        tok2m[drop, tok_i, :] = 0.0
            pres, tok_img = pres2m, tok2m

        if not return_aux:
            return tok_img, pres

        aux: Dict[str, torch.Tensor] = {
            "loc_pt_logits": loc_pt_logits,
            "loc_ln_logits": loc_ln_logits,
            "pt_prob": pt_prob,
            "ln_prob": ln_prob,
            "pt_used": pt_used,
            "ln_used": ln_used,
            "pt_presence_logits": pt_presence_logits,
            "ln_presence_logits": ln_presence_logits,
        }
        return tok_img, pres, aux
