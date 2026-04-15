"""SwinUNETR token MoE discrete-time survival model for TriFuseSurv."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from trifusesurv.models.swinunetr_shared_roi_token_backbone import (
    ContourAwareROITokenBackbone,
)

SURVIVAL_ENDPOINTS = ("OS", "DSS", "DFS")


# ---------------------------------------------------------------------------
# Gate penalties
# ---------------------------------------------------------------------------
def gate_entropy_penalty_presence(
    gate: torch.Tensor, presence: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    g = gate.float()
    p = presence.float()
    gp = g * p
    Z = gp.sum(dim=1, keepdim=True).clamp(min=eps)
    gp = gp / Z
    ent = -(gp * torch.log(gp + eps)).sum(dim=1)
    m_av = p.sum(dim=1).clamp(min=1.0)
    max_ent = torch.log(m_av)
    return (max_ent - ent).mean()


def gate_load_balance_penalty_presence(
    gate: torch.Tensor, presence: torch.Tensor, eps: float = 1e-8,
) -> torch.Tensor:
    g = gate.float()
    p = presence.float()
    denom_raw = p.sum(dim=0)
    active = (denom_raw > eps).float()
    denom = denom_raw.clamp(min=1.0)
    usage = (g * p).sum(dim=0) / denom
    M = active.sum().clamp(min=1.0)
    target = active / M
    pen = (((usage - target) ** 2) * active).sum() / M
    return pen


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SwinUNETRTokenMoEDiscrete(nn.Module):
    """Multimodal discrete-time survival model."""

    def __init__(
        self,
        *,
        num_time_bins: int,
        time_bin_width_days: float,
        fused_dim: int,
        backbone_cfg: Dict[str, Any],
        clinical_dim: int,
        radiomics_dim: int,
        expert_dropout_p: float,
        proj_dropout_p: float,
        attn_dropout_p: float,
        gate_dropout_p: float,
        surv_dropout_p: float,
        clinical_noise_std: float,
        radiomics_noise_std: float,
        modality_dropout_clin_p: float,
        modality_dropout_rad_p: float,
        img_proj_hidden_dim: int = 0,
        img_tok_ffn_hidden_dim: int = 0,
        img_post_hidden_dim: int = 0,
        img_attn_heads: int = 4,
        gate_hidden_dim: int = 0,
        rad_hidden_dim: int = 0,
        rad_proj_dropout_p: float = 0.50,
        nan_guard: bool = False,
    ):
        super().__init__()
        self.num_time_bins = int(num_time_bins)
        self.time_bin_width_days = float(time_bin_width_days)
        self.fused_dim = int(fused_dim)
        self.nan_guard = bool(nan_guard)

        backbone_cfg = dict(backbone_cfg)
        self.image_encoder_mode = str(backbone_cfg.pop("image_encoder_mode", "contour_aware")).strip().lower()
        if self.image_encoder_mode != "contour_aware":
            raise ValueError(f"Unsupported image_encoder_mode for compact package: {self.image_encoder_mode}")
        self.img_backbone = ContourAwareROITokenBackbone(**backbone_cfg)
        self.num_experts = int(self.img_backbone.num_tokens)

        self.expert_dropout_p = float(expert_dropout_p)

        proj_drop = float(proj_dropout_p)
        img_proj_hidden_dim = int(img_proj_hidden_dim) if int(img_proj_hidden_dim) > 0 else int(2 * self.fused_dim)
        img_tok_ffn_hidden_dim = int(img_tok_ffn_hidden_dim) if int(img_tok_ffn_hidden_dim) > 0 else int(2 * self.fused_dim)
        img_post_hidden_dim = int(img_post_hidden_dim) if int(img_post_hidden_dim) > 0 else int(2 * self.fused_dim)

        self.fuse_projs = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.img_backbone.out_dim),
                nn.Linear(self.img_backbone.out_dim, img_proj_hidden_dim),
                nn.GELU(),
                nn.Dropout(p=proj_drop),
                nn.Linear(img_proj_hidden_dim, self.fused_dim),
            )
            for _ in range(self.num_experts)
        ])

        self.img_tok_ln = nn.LayerNorm(self.fused_dim)

        img_attn_heads = int(img_attn_heads)
        if img_attn_heads < 1:
            img_attn_heads = 1
        if (self.fused_dim % img_attn_heads) != 0:
            print(f"[WARN] fused_dim={self.fused_dim} not divisible by img_attn_heads={img_attn_heads}; forcing 1 head.")
            img_attn_heads = 1

        self.img_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim,
            num_heads=img_attn_heads,
            batch_first=True,
            dropout=float(attn_dropout_p),
        )

        self.img_tok_ffn = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Linear(self.fused_dim, img_tok_ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=proj_drop),
            nn.Linear(img_tok_ffn_hidden_dim, self.fused_dim),
            nn.Dropout(p=proj_drop),
        )

        self.img_post_mlp = nn.Sequential(
            nn.LayerNorm(self.fused_dim),
            nn.Linear(self.fused_dim, img_post_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=proj_drop),
            nn.Linear(img_post_hidden_dim, self.fused_dim),
            nn.Dropout(p=proj_drop),
        )

        gate_hidden_dim = int(gate_hidden_dim) if int(gate_hidden_dim) > 0 else int(self.fused_dim)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self.fused_dim * self.num_experts),
            nn.Dropout(p=float(gate_dropout_p)),
            nn.Linear(self.fused_dim * self.num_experts, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=float(gate_dropout_p)),
            nn.Linear(gate_hidden_dim, self.num_experts),
        )

        self.clin_noise_std = float(clinical_noise_std)
        self.rad_noise_std = float(radiomics_noise_std)
        self.drop_clin_p = float(modality_dropout_clin_p)
        self.drop_rad_p = float(modality_dropout_rad_p)

        self.use_clin = int(clinical_dim) > 0
        self.clin_proj = nn.Sequential(
            nn.LayerNorm(int(clinical_dim)),
            nn.Dropout(p=0.2),
            nn.Linear(int(clinical_dim), self.fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
        ) if self.use_clin else None

        self.use_rad = int(radiomics_dim) > 0
        rad_hidden_dim = int(rad_hidden_dim) if int(rad_hidden_dim) > 0 else int(max(512, 2 * self.fused_dim))
        rad_drop = float(rad_proj_dropout_p)
        self.rad_proj = nn.Sequential(
            nn.LayerNorm(int(radiomics_dim)),
            nn.Dropout(p=0.2),
            nn.Linear(int(radiomics_dim), rad_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=rad_drop),
            nn.Linear(rad_hidden_dim, rad_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=rad_drop),
            nn.Linear(rad_hidden_dim, self.fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=rad_drop),
        ) if self.use_rad else None

        surv_in = self.fused_dim + (self.fused_dim if self.use_clin else 0) + (self.fused_dim if self.use_rad else 0)
        self.surv_head_input_dim = int(surv_in)
        self.surv_hidden_dim = 256
        self.surv_dropout_p = float(surv_dropout_p)
        self.surv_heads = nn.ModuleDict({
            endpoint: self._make_survival_head()
            for endpoint in SURVIVAL_ENDPOINTS
        })

    def _make_survival_head(self) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(self.surv_head_input_dim),
            nn.Linear(self.surv_head_input_dim, self.surv_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.surv_dropout_p),
            nn.Linear(self.surv_hidden_dim, self.num_time_bins),
        )

    def enable_mask_patch_embed_training(self, verbose: bool = True):
        self.img_backbone.enable_mask_patch_embed_training(verbose=verbose)

    def hazards_to_risk(self, hazards_logits: torch.Tensor, horizon_days: float) -> torch.Tensor:
        bw = float(getattr(self, "time_bin_width_days", 0.0))
        if bw <= 0.0:
            raise ValueError(f"time_bin_width_days must be > 0, got {bw}")
        t = float(horizon_days)
        B, K = hazards_logits.shape[0], hazards_logits.shape[1]
        if K <= 0 or t <= 0.0:
            return hazards_logits.new_zeros((B,), dtype=torch.float32)
        hazards = torch.sigmoid(hazards_logits.float()).clamp(1e-7, 1.0 - 1e-7)
        max_covered = K * bw
        if t >= max_covered:
            logS = torch.log1p(-hazards).sum(dim=1)
            return (1.0 - torch.exp(logS)).clamp(0.0, 1.0)
        k = int(math.floor(t / bw))
        k = max(0, min(k, K - 1))
        within = t - (k * bw)
        frac = within / bw
        log1m = torch.log1p(-hazards)
        cum = torch.cumsum(log1m, dim=1)
        if k == 0:
            logS_t = log1m[:, 0] * float(t / bw)
        else:
            logS_prev = cum[:, k - 1]
            logS_t = logS_prev + log1m[:, k] * float(frac)
        return (1.0 - torch.exp(logS_t)).clamp(0.0, 1.0)

    def _nan_check(self, x: torch.Tensor, name: str):
        if self.nan_guard and (not torch.isfinite(x).all().item()):
            bad = (~torch.isfinite(x)).sum().item()
            raise RuntimeError(f"[NaNGuard] tensor '{name}' has {bad} non-finite entries.")

    def forward(
        self,
        x_img: torch.Tensor,
        clinical: Optional[torch.Tensor],
        radiomics: Optional[torch.Tensor],
        mask_pt: Optional[torch.Tensor] = None,
        mask_ln: Optional[torch.Tensor] = None,
        teacher_force_alpha: float = 0.0,
        return_gate: bool = False,
        return_aux: bool = False,
    ):
        aux = None
        can_return_aux = bool(return_aux)
        bb_out = self.img_backbone(
            x_img,
            mask_pt=mask_pt,
            mask_ln=mask_ln,
            teacher_force_alpha=float(teacher_force_alpha),
            return_aux=can_return_aux,
        )

        if can_return_aux:
            tok, pres, aux = bb_out
        else:
            tok, pres = bb_out
        B = tok.size(0)
        pres_bool = pres.to(torch.bool)

        fused_list = [self.fuse_projs[i](tok[:, i, :]) for i in range(self.num_experts)]
        stacked = torch.stack(fused_list, dim=1)  # (B,E,D)
        stacked = stacked * pres_bool.unsqueeze(-1).to(stacked.dtype)

        # Guard against all-masked attention (can produce NaNs)
        pres_attn = pres_bool
        all_abs = ~pres_attn.any(dim=1)
        if all_abs.any():
            pres_attn = pres_attn.clone()
            pres_attn[all_abs, 0] = True

        q = self.img_tok_ln(stacked)
        attn_out, _ = self.img_attn(q, q, q, key_padding_mask=(~pres_attn))
        attn_out = attn_out * pres_attn.unsqueeze(-1).to(attn_out.dtype)
        attn_out = (attn_out + stacked) * pres_attn.unsqueeze(-1).to(attn_out.dtype)
        attn_out = (attn_out + self.img_tok_ffn(attn_out)) * pres_attn.unsqueeze(-1).to(attn_out.dtype)

        self._nan_check(attn_out, "attn_out")

        # Expert dropout
        pres_eff = pres_attn
        if self.training and self.expert_dropout_p > 0:
            keep = pres_eff.clone()
            drop = (torch.rand(B, self.num_experts, device=attn_out.device) < self.expert_dropout_p) & keep
            keep2 = keep & (~drop)
            none_active = ~keep2.any(dim=1)
            if none_active.any():
                keep2 = keep2.clone()
                keep2[none_active, 0] = True
            attn_out = attn_out * keep2.unsqueeze(-1).to(attn_out.dtype)
            pres_eff = keep2

        # Gate
        gate_logits = self.gate_mlp(attn_out.reshape(B, -1).float()).float()
        neg_inf = torch.finfo(gate_logits.dtype).min
        gate_logits = gate_logits.masked_fill(~pres_eff, neg_inf)
        all_abs2 = ~pres_eff.any(dim=1)
        if all_abs2.any():
            gate_logits = gate_logits.clone()
            gate_logits[all_abs2, 0] = 0.0
        gate = torch.softmax(gate_logits, dim=1).to(attn_out.dtype)

        self._nan_check(gate, "gate")

        fused_img = (attn_out * gate.unsqueeze(-1)).sum(dim=1)
        fused_img = fused_img + self.img_post_mlp(fused_img)

        chunks = [fused_img]

        if self.use_clin and clinical is not None and clinical.numel() > 0:
            c = clinical.to(fused_img.device)
            if self.training and self.drop_clin_p > 0:
                dropc = (torch.rand(c.size(0), 1, device=c.device) < self.drop_clin_p).to(c.dtype)
                c = c * (1.0 - dropc)
            if self.training and self.clin_noise_std > 0:
                c = c + self.clin_noise_std * torch.randn_like(c)
            chunks.append(self.clin_proj(c))

        if self.use_rad and radiomics is not None and radiomics.numel() > 0:
            r = radiomics.to(fused_img.device)
            if self.training and self.drop_rad_p > 0:
                dropr = (torch.rand(r.size(0), 1, device=r.device) < self.drop_rad_p).to(r.dtype)
                r = r * (1.0 - dropr)
            if self.training and self.rad_noise_std > 0:
                r = r + self.rad_noise_std * torch.randn_like(r)
            chunks.append(self.rad_proj(r))

        h = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        logits = {}
        for endpoint, head in self.surv_heads.items():
            ep_logits = head(h)
            self._nan_check(ep_logits, f"{endpoint}_logits_pre_nan_to_num")
            logits[endpoint] = torch.nan_to_num(ep_logits, nan=0.0, posinf=0.0, neginf=0.0)

        if return_gate and can_return_aux:
            return logits, gate.float(), pres_eff, aux
        if return_gate:
            return logits, gate.float(), pres_eff
        if can_return_aux:
            return logits, aux
        return logits
