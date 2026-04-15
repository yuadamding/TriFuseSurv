#!/usr/bin/env python3
"""Joint contour-aware multimodal survival training for TriFuseSurv.

Recommended workflow:
- CT-only shared SwinUNETR encoder
- internal PT/LN localization heads
- ROI tokens built from soft predicted masks
- survival plus localization losses in one graph
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import math
import json
import contextlib
import argparse
from typing import Tuple, Dict, List, Optional, Any, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parameter import UninitializedParameter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR

from trifusesurv.models.swinunetr_backbone_utils import load_swinunetr_pretrained
from trifusesurv.models.survival_model import (
    SwinUNETRTokenMoEDiscrete,
    SURVIVAL_ENDPOINTS,
    gate_entropy_penalty_presence,
    gate_load_balance_penalty_presence,
)
from trifusesurv.models.lora import (
    inject_lora_into_module,
    freeze_all_params,
    mark_only_lora_trainable,
    count_trainable,
    is_lora_param_name,
)
from trifusesurv.utils import survival as H
from trifusesurv.utils.clinical import (
    ClinicalEncoder,
    DEFAULT_CLINICAL_COLS,
    ENDPOINT_MAP,
)
from trifusesurv.utils.radiomics import RadiomicsEncoder
from trifusesurv.utils.data import PreprocessedContourAwareDataset
from trifusesurv.utils.data import resolve_preprocessed_case_path


# =============================================================================
# Seed / device / AMP helpers (set_seed, seed_worker from utils.survival)
# =============================================================================
set_seed = H.set_seed
seed_worker = H.seed_worker

ENDPOINT_TO_INDEX = {name: idx for idx, name in enumerate(SURVIVAL_ENDPOINTS)}
MULTITASK_TIME_COLS = tuple(ENDPOINT_MAP[name][0] for name in SURVIVAL_ENDPOINTS)
MULTITASK_EVENT_COLS = tuple(ENDPOINT_MAP[name][1] for name in SURVIVAL_ENDPOINTS)


def _configure_stdio_line_buffering():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except Exception:
                pass


def _log(msg: str):
    print(msg, flush=True)


def parse_device(device_str: str) -> torch.device:
    dev = str(device_str).strip().lower()
    if dev == "" or dev == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device but CUDA is not available.")
        return torch.device(dev)
    raise ValueError(f"--device must be cpu|cuda|cuda:N (or empty), got: {device_str}")


def bind_cuda_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.set_device(int(device.index) if device.index is not None else 0)


def make_amp(device: torch.device, enabled: bool):
    amp_enabled = bool(enabled and device.type == "cuda")
    if not amp_enabled:
        return None, (lambda: contextlib.nullcontext())
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=True)
        return scaler, autocast_ctx
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        autocast_ctx = lambda: torch.cuda.amp.autocast(enabled=True)
        return scaler, autocast_ctx


def _find_first_existing_path(meta: pd.DataFrame, col: str, *, id_col: str = "patient_id", data_root: str = "") -> Optional[str]:
    if col not in meta.columns:
        return None
    ids = meta[id_col].astype(str).tolist() if id_col in meta.columns else [""] * len(meta)
    for pid, p in zip(ids, meta[col].astype(str).tolist()):
        p_resolved = resolve_preprocessed_case_path(p, data_root=data_root, patient_id=pid)
        if p_resolved and os.path.isfile(p_resolved):
            return p_resolved
    return None


def _read_nii_shape(path: str) -> Tuple[int, int, int]:
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    return tuple(int(x) for x in arr.shape)


def resolve_img_size_against_data(
    meta: pd.DataFrame,
    ct_col: str,
    img_size_arg: Sequence[int],
    *,
    id_col: str = "patient_id",
    data_root: str = "",
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    arg = tuple(int(x) for x in img_size_arg)
    p0 = _find_first_existing_path(meta, ct_col, id_col=id_col, data_root=data_root)
    if p0 is None:
        print(f"[IMGCFG][WARN] Could not find any existing CT path to validate --img_size. Using as-is: {arg}")
        return arg, arg

    shp = _read_nii_shape(p0)  # (D,H,W)
    if shp == arg:
        return arg, shp
    if shp == tuple(reversed(arg)):
        print(
            f"[IMGCFG][WARN] Data shape is {shp} (D,H,W) but --img_size={arg}. "
            f"Interpreting --img_size as (H,W,D) and flipping to (D,H,W)={shp}."
        )
        return shp, shp

    print(
        f"[IMGCFG][WARN] Data CT shape is {shp} (D,H,W) but --img_size={arg}. "
        f"Overriding to the data shape for stage-2 training."
    )
    return shp, shp


# LoRA classes/functions imported from trifusesurv.models.lora


# =============================================================================
# Contour-warmstart cfg alignment
# =============================================================================
def read_contour_warmstart_backbone_cfg(ckpt_path: str) -> Dict[str, Any]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_args = ck.get("args", {}) if isinstance(ck.get("args", {}), dict) else {}

    img_size = ck.get("img_size_dhw", None)
    if img_size is None:
        img_size = ck_args.get("img_size", None)
    if img_size is None:
        img_size = (128, 256, 256)

    cfg = dict(
        img_size=tuple(int(x) for x in img_size),
        feature_size=int(ck_args.get("feature_size", 96)),
        depths=tuple(int(x) for x in ck_args.get("depths", (2, 2, 18, 2))),
        num_heads=tuple(int(x) for x in ck_args.get("num_heads", (3, 6, 12, 24))),
        drop_rate=float(ck_args.get("drop_rate", 0.0)),
        attn_drop_rate=float(ck_args.get("attn_drop_rate", 0.0)),
        dropout_path_rate=float(ck_args.get("dropout_path_rate", 0.0)),
        use_checkpoint=bool(ck_args.get("use_checkpoint", True)),
    )
    return cfg


def align_backbone_cfg_to_contour_warmstart(args):
    contour_ckpt = _resolve_existing_contour_warmstart_ckpt_for_cfg(args)
    cfg = read_contour_warmstart_backbone_cfg(contour_ckpt) if (contour_ckpt and os.path.isfile(contour_ckpt)) else None
    if cfg is None:
        print("[SWINCFG] No contour-aware warm-start ckpt found for cfg alignment; using CLI Swin cfg.")
        return args

    want_img = tuple(int(x) for x in cfg["img_size"])
    want_fs = int(cfg["feature_size"])
    want_depths = tuple(int(x) for x in cfg["depths"])
    want_heads = tuple(int(x) for x in cfg["num_heads"])

    cur_img = tuple(int(x) for x in args.img_size)
    cur_fs = int(args.feature_size)
    cur_depths = tuple(int(x) for x in args.depths)
    cur_heads = tuple(int(x) for x in args.num_heads)

    if cur_img != want_img:
        print(f"[SWINCFG][WARN] CLI --img_size {cur_img} != contour-aware warm-start {want_img}. Overriding to warm-start cfg.")
        args.img_size = list(want_img)
    if (cur_fs, cur_depths, cur_heads) != (want_fs, want_depths, want_heads):
        print("[SWINCFG][WARN] CLI Swin cfg != contour-aware warm-start cfg. Overriding to match warm-start.")
        print(f"  CLI : feature_size={cur_fs} depths={cur_depths} num_heads={cur_heads}")
        print(f"  CKPT: feature_size={want_fs} depths={want_depths} num_heads={want_heads}")

    args.feature_size = want_fs
    args.depths = list(want_depths)
    args.num_heads = list(want_heads)

    args.drop_rate = float(cfg.get("drop_rate", args.drop_rate))
    args.attn_drop_rate = float(cfg.get("attn_drop_rate", args.attn_drop_rate))
    args.dropout_path_rate = float(cfg.get("dropout_path_rate", args.dropout_path_rate))

    if bool(cfg.get("use_checkpoint", True)) and (not bool(args.use_checkpoint)):
        args.use_checkpoint = True

    print("[SWINCFG] aligned to contour-aware warm-start:")
    print(
        f"  img_size={tuple(args.img_size)} feature_size={args.feature_size} "
        f"depths={tuple(args.depths)} heads={tuple(args.num_heads)}"
    )
    print(
        f"  drop_rate={args.drop_rate} attn_drop_rate={args.attn_drop_rate} "
        f"drop_path={args.dropout_path_rate} use_checkpoint={args.use_checkpoint}"
    )
    if contour_ckpt:
        print(f"  contour_warmstart_ckpt_for_cfg={contour_ckpt}")
    return args


# =============================================================================
# Splits
# =============================================================================
def _read_id_list(path: str) -> List[str]:
    with open(path, "r") as f:
        ids = [ln.strip() for ln in f.read().splitlines()]
    return [x for x in ids if x]


def load_precomputed_splits(cv_folds: int, *, splits_dir: str = "", splits_csv: str = "") -> Dict[int, Dict[str, List[str]]]:
    if bool(splits_dir) == bool(splits_csv):
        raise ValueError("Provide exactly one of --splits_dir or --splits_csv.")
    out: Dict[int, Dict[str, List[str]]] = {}

    if splits_dir:
        for f in range(int(cv_folds)):
            fold_dir = os.path.join(splits_dir, f"fold_{f:02d}")
            tr = _read_id_list(os.path.join(fold_dir, "train_ids.txt"))
            va = _read_id_list(os.path.join(fold_dir, "val_ids.txt"))
            te = _read_id_list(os.path.join(fold_dir, "test_ids.txt"))
            out[f] = {"train": tr, "val": va, "test": te}
        return out

    df = pd.read_csv(splits_csv, dtype={"patient_id": str, "split": str})
    need = {"patient_id", "fold", "split"}
    if not need.issubset(df.columns):
        raise ValueError(f"--splits_csv must contain {sorted(need)}; got {list(df.columns)}")
    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["fold"] = pd.to_numeric(df["fold"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.lower()

    for f in range(int(cv_folds)):
        dff = df[df["fold"] == f]
        if dff.empty:
            raise ValueError(f"--splits_csv has no rows for fold={f}")
        out[f] = {
            "train": dff.loc[dff["split"] == "train", "patient_id"].tolist(),
            "val":   dff.loc[dff["split"] == "val",   "patient_id"].tolist(),
            "test":  dff.loc[dff["split"] == "test",  "patient_id"].tolist(),
        }
    return out


def _assert_split_disjoint(fold: int, split: Dict[str, List[str]]):
    tr = set(map(str, split.get("train", [])))
    va = set(map(str, split.get("val", [])))
    te = set(map(str, split.get("test", [])))
    bad = (tr & va) | (tr & te) | (va & te)
    if bad:
        raise RuntimeError(f"[SPLIT] fold {fold:02d} overlap among train/val/test: {sorted(list(bad))[:50]}")


# =============================================================================
# Preprocess-output I/O
# =============================================================================
def read_nii(path: str, dtype=np.float32) -> np.ndarray:
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (D,H,W)
    arr = np.asarray(arr, dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# rand_flip_3d and rand_intensity are now imported from utils.data


# ClinicalEncoder is now imported from utils.clinical


# _pad_or_trunc_1d and RadiomicsEncoder are now imported from utils.radiomics


# gate_entropy_penalty_presence and gate_load_balance_penalty_presence are now imported from models.survival_model


# SwinUNETRTokenMoEDiscrete is now imported from models.survival_model


# =============================================================================
# Lazy materialization
# =============================================================================
def _unpack_surv_batch(batch):
    if len(batch) == 6:
        x, t, e, clin, rad, pid = batch
        return dict(x=x, mask_pt=None, mask_ln=None, t=t, e=e, t_all=t[:, None], e_all=e[:, None], clin=clin, rad=rad, pid=pid)
    if len(batch) == 8:
        x, mask_pt, mask_ln, t, e, clin, rad, pid = batch
        return dict(x=x, mask_pt=mask_pt, mask_ln=mask_ln, t=t, e=e, t_all=t[:, None], e_all=e[:, None], clin=clin, rad=rad, pid=pid)
    if len(batch) == 10:
        x, mask_pt, mask_ln, t, e, t_all, e_all, clin, rad, pid = batch
        return dict(x=x, mask_pt=mask_pt, mask_ln=mask_ln, t=t, e=e, t_all=t_all, e_all=e_all, clin=clin, rad=rad, pid=pid)
    raise RuntimeError(f"[BATCH] Unexpected batch structure of length {len(batch)}")


def _to_optional_device_tensor(x: Optional[torch.Tensor], device: torch.device):
    if x is None:
        return None
    return x.to(device, non_blocking=True)


def _teacher_force_alpha_for_epoch(epoch: int, args) -> float:
    epochs = int(max(0, getattr(args, "teacher_force_epochs", 0)))
    start = float(getattr(args, "teacher_force_start", 1.0))
    end = float(getattr(args, "teacher_force_end", 0.0))
    if epochs <= 0:
        return 0.0
    if epochs == 1:
        return float(start)
    if int(epoch) > epochs:
        return float(end)
    frac = float(max(0.0, min(1.0, (int(epoch) - 1) / float(epochs - 1))))
    return float(start + frac * (end - start))


def _primary_endpoint_index(endpoint: str) -> int:
    endpoint = str(endpoint).strip().upper()
    if endpoint not in ENDPOINT_TO_INDEX:
        raise KeyError(f"Unknown endpoint: {endpoint}")
    return int(ENDPOINT_TO_INDEX[endpoint])


def _extract_endpoint_logits(logits_by_endpoint, endpoint: str) -> torch.Tensor:
    if isinstance(logits_by_endpoint, dict):
        key = str(endpoint).strip().upper()
        if key not in logits_by_endpoint:
            raise KeyError(f"Missing logits for endpoint {key}. Available: {sorted(logits_by_endpoint.keys())}")
        return logits_by_endpoint[key]
    return logits_by_endpoint


def _valid_survival_mask(times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    t = times.float()
    e = events.float()
    return torch.isfinite(t) & torch.isfinite(e) & (t > 0.0) & ((e == 0.0) | (e == 1.0))


def _valid_survival_mask_frame(df: pd.DataFrame, time_col: str, event_col: str) -> pd.Series:
    times = pd.to_numeric(df[time_col], errors="coerce")
    events = pd.to_numeric(df[event_col], errors="coerce")
    return times.notna() & events.notna() & (times > 0.0) & events.isin([0, 1])


def _endpoint_survival_losses(
    logits_by_endpoint,
    t_all: torch.Tensor,
    e_all: torch.Tensor,
    *,
    time_bin_width_days: float,
    num_time_bins: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    losses: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    available_heads = int(t_all.shape[1]) if t_all.ndim >= 2 else 1

    for idx, endpoint in enumerate(SURVIVAL_ENDPOINTS):
        if idx >= available_heads:
            continue
        logits = _extract_endpoint_logits(logits_by_endpoint, endpoint).float()
        if logits.ndim != 2:
            raise RuntimeError(f"[MTL] Expected 2D logits for {endpoint}, got shape={tuple(logits.shape)}")
        t_ep = t_all[:, idx].to(logits.device)
        e_ep = e_all[:, idx].to(logits.device)
        valid = _valid_survival_mask(t_ep, e_ep)
        counts[endpoint] = int(valid.sum().item())
        if not bool(valid.any().item()):
            continue
        losses[endpoint] = H.discrete_time_nll_loss(
            logits[valid],
            t_ep[valid],
            e_ep[valid],
            float(time_bin_width_days),
            int(num_time_bins),
        )

    return losses, counts


def _weighted_multitask_mean(
    values_by_endpoint: Dict[str, torch.Tensor],
    counts_by_endpoint: Dict[str, int],
    *,
    primary_endpoint: str,
    primary_weight: float,
    aux_weight: float,
    ref_tensor: torch.Tensor,
) -> torch.Tensor:
    total = None
    total_weight = 0.0
    primary_key = str(primary_endpoint).strip().upper()
    for endpoint, value in values_by_endpoint.items():
        count = float(counts_by_endpoint.get(endpoint, 0))
        if count <= 0.0:
            continue
        endpoint_weight = float(primary_weight if endpoint == primary_key else aux_weight)
        if endpoint_weight <= 0.0:
            continue
        combined_weight = endpoint_weight * count
        total = (value * combined_weight) if total is None else (total + (value * combined_weight))
        total_weight += combined_weight

    if total is None or total_weight <= 0.0:
        return ref_tensor.new_tensor(float("nan"))
    return total / float(total_weight)


def _multitask_survival_loss(
    logits_by_endpoint,
    t_all: torch.Tensor,
    e_all: torch.Tensor,
    *,
    primary_endpoint: str,
    primary_surv_loss_weight: float,
    aux_surv_loss_weight: float,
    time_bin_width_days: float,
    num_time_bins: int,
) -> torch.Tensor:
    primary_key = str(primary_endpoint).strip().upper()
    endpoint_losses, _ = _endpoint_survival_losses(
        logits_by_endpoint,
        t_all=t_all,
        e_all=e_all,
        time_bin_width_days=float(time_bin_width_days),
        num_time_bins=int(num_time_bins),
    )
    _, endpoint_counts = _endpoint_survival_losses(
        logits_by_endpoint,
        t_all=t_all,
        e_all=e_all,
        time_bin_width_days=float(time_bin_width_days),
        num_time_bins=int(num_time_bins),
    )
    ref = _extract_endpoint_logits(logits_by_endpoint, primary_key)
    return _weighted_multitask_mean(
        endpoint_losses,
        endpoint_counts,
        primary_endpoint=primary_key,
        primary_weight=float(primary_surv_loss_weight),
        aux_weight=float(aux_surv_loss_weight),
        ref_tensor=ref,
    )


def _resize_mask_target(mask: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(int(x) for x in mask.shape[2:]) == tuple(int(x) for x in size):
        return mask.clamp(0, 1)
    return F.interpolate(mask.float(), size=size, mode="nearest").clamp(0, 1)


def _soft_dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    numer = 2.0 * (probs * target).sum(dim=(1, 2, 3, 4))
    denom = probs.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (numer + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _localization_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, bce_weight: float, dice_weight: float) -> torch.Tensor:
    loss = logits.new_tensor(0.0)
    if float(bce_weight) > 0.0:
        loss = loss + float(bce_weight) * F.binary_cross_entropy_with_logits(logits, target)
    if float(dice_weight) > 0.0:
        loss = loss + float(dice_weight) * _soft_dice_loss_from_logits(logits, target)
    return loss


def materialize_lazy_modules(model: nn.Module, loader: DataLoader, device: torch.device, autocast_ctx):
    has_lazy = any(isinstance(p, UninitializedParameter) for p in model.parameters())
    if not has_lazy:
        return
    _log("[init] materializing lazy modules from first batch...")
    batch = next(iter(loader), None)
    if batch is None:
        _log("[init][warn] loader yielded no batch; skipping lazy-module materialization.")
        return
    payload = _unpack_surv_batch(batch)
    x = payload["x"].to(device, non_blocking=True)
    clin = payload["clin"].to(device) if (payload["clin"] is not None and payload["clin"].numel() > 0) else None
    rad = payload["rad"].to(device) if (payload["rad"] is not None and payload["rad"].numel() > 0) else None
    mask_pt = _to_optional_device_tensor(payload["mask_pt"], device)
    mask_ln = _to_optional_device_tensor(payload["mask_ln"], device)
    model.eval()
    with torch.no_grad():
        with autocast_ctx():
            _ = model(x, clin, rad, mask_pt=mask_pt, mask_ln=mask_ln, teacher_force_alpha=0.0, return_gate=False)
    _log("[init] lazy modules materialized.")


# =============================================================================
# Optimizer groups (LoRA-aware) + disjointness checks
# =============================================================================
def _split_decay_no_decay(named_params: List[Tuple[str, torch.nn.Parameter]]):
    decay, no_decay = [], []
    for n, p in named_params:
        if isinstance(p, UninitializedParameter):
            continue
        if not p.requires_grad:
            continue
        ln = n.lower()
        if p.ndim <= 1 or ln.endswith(".bias") or ("norm" in ln) or ("layernorm" in ln):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


# _is_lora_param_name is now imported as is_lora_param_name from models.lora


def _assert_param_groups_disjoint(groups: List[Dict[str, Any]]):
    seen = set()
    dup = 0
    total = 0
    for g in groups:
        for p in g.get("params", []):
            total += 1
            pid = id(p)
            if pid in seen:
                dup += 1
            seen.add(pid)
    if dup > 0:
        raise RuntimeError(f"[OPT] Param groups are not disjoint: duplicates={dup} / total_params_in_groups={total}")


def make_param_groups(
    mm: SwinUNETRTokenMoEDiscrete,
    *,
    lr_backbone: float,
    lr_lora: float,
    lr_head: float,
    wd_backbone: float,
    wd_lora: float,
    wd_head: float,
    wd_clin: float,
    wd_rad: float,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []

    enc_lora_named: List[Tuple[str, torch.nn.Parameter]] = []
    enc_nonlora_named: List[Tuple[str, torch.nn.Parameter]] = []

    encoder_pairs = _iter_image_encoder_backbones(mm.img_backbone)
    encoder_prefixes = []
    for prefix, bb in encoder_pairs:
        encoder_prefixes.append(f"{prefix}.")
        for n, p in bb.named_parameters():
            full = f"img_backbone.{prefix}.{n}"
            if not p.requires_grad:
                continue
            if is_lora_param_name(full):
                enc_lora_named.append((full, p))
            else:
                enc_nonlora_named.append((full, p))

    enc_d, enc_n = _split_decay_no_decay(enc_nonlora_named)
    if enc_d:
        groups.append({"params": enc_d, "lr": float(lr_backbone), "weight_decay": float(wd_backbone)})
    if enc_n:
        groups.append({"params": enc_n, "lr": float(lr_backbone), "weight_decay": 0.0})

    lora_d, lora_n = _split_decay_no_decay(enc_lora_named)
    if lora_d:
        groups.append({"params": lora_d, "lr": float(lr_lora), "weight_decay": float(wd_lora)})
    if lora_n:
        groups.append({"params": lora_n, "lr": float(lr_lora), "weight_decay": 0.0})

    head_named: List[Tuple[str, torch.nn.Parameter]] = []
    for n, p in mm.img_backbone.named_parameters():
        if any(n.startswith(pref) for pref in encoder_prefixes):
            continue
        head_named.append((f"img_backbone.{n}", p))
    for n, p in mm.fuse_projs.named_parameters():
        head_named.append((f"fuse_projs.{n}", p))
    for n, p in mm.img_tok_ln.named_parameters():
        head_named.append((f"img_tok_ln.{n}", p))
    for n, p in mm.img_attn.named_parameters():
        head_named.append((f"img_attn.{n}", p))
    for n, p in mm.img_tok_ffn.named_parameters():
        head_named.append((f"img_tok_ffn.{n}", p))
    for n, p in mm.img_post_mlp.named_parameters():
        head_named.append((f"img_post_mlp.{n}", p))
    for n, p in mm.gate_mlp.named_parameters():
        head_named.append((f"gate_mlp.{n}", p))
    for n, p in mm.surv_heads.named_parameters():
        head_named.append((f"surv_heads.{n}", p))

    head_d, head_n = _split_decay_no_decay(head_named)
    if head_d:
        groups.append({"params": head_d, "lr": float(lr_head), "weight_decay": float(wd_head)})
    if head_n:
        groups.append({"params": head_n, "lr": float(lr_head), "weight_decay": 0.0})

    if mm.clin_proj is not None:
        clin_named = [(f"clin_proj.{n}", p) for n, p in mm.clin_proj.named_parameters()]
        clin_d, clin_n = _split_decay_no_decay(clin_named)
        if clin_d:
            groups.append({"params": clin_d, "lr": float(lr_head), "weight_decay": float(wd_clin)})
        if clin_n:
            groups.append({"params": clin_n, "lr": float(lr_head), "weight_decay": 0.0})

    if mm.rad_proj is not None:
        rad_named = [(f"rad_proj.{n}", p) for n, p in mm.rad_proj.named_parameters()]
        rad_d, rad_n = _split_decay_no_decay(rad_named)
        if rad_d:
            groups.append({"params": rad_d, "lr": float(lr_head), "weight_decay": float(wd_rad)})
        if rad_n:
            groups.append({"params": rad_n, "lr": float(lr_head), "weight_decay": 0.0})

    if not groups:
        raise RuntimeError("[OPT] No trainable parameters found.")

    _assert_param_groups_disjoint(groups)
    return groups


# =============================================================================
# Evaluation helpers
# =============================================================================
@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    endpoint: str,
    risk_horizon_days: float,
    time_bin_width_days: float,
    eval_times_days: Sequence[float],
    dca_thresholds: Sequence[float],
    autocast_ctx,
) -> Dict[str, float]:
    model.eval()
    mm = model.module if isinstance(model, nn.DataParallel) else model

    all_t, all_e, all_r, all_h = [], [], [], []

    nll_sum = 0.0
    nll_n = 0

    for batch in loader:
        payload = _unpack_surv_batch(batch)
        x = payload["x"].to(device, non_blocking=True)
        t = payload["t"]
        e = payload["e"]
        t_dev = t.to(device, non_blocking=True)
        e_dev = e.to(device, non_blocking=True)

        clin = payload["clin"].to(device) if (payload["clin"] is not None and payload["clin"].numel() > 0) else None
        rad = payload["rad"].to(device) if (payload["rad"] is not None and payload["rad"].numel() > 0) else None

        with autocast_ctx():
            logits = model(x, clin, rad, teacher_force_alpha=0.0, return_gate=False)
            logits = _extract_endpoint_logits(logits, endpoint)

        lf = logits.float()
        K = int(lf.shape[1])
        batch_nll = H.discrete_time_nll_loss(lf, t_dev, e_dev, float(time_bin_width_days), K)
        bs = int(t.shape[0])
        nll_sum += float(batch_nll.detach().cpu().item()) * bs
        nll_n += bs

        risks = mm.hazards_to_risk(lf, horizon_days=float(risk_horizon_days)).detach().cpu().numpy()
        haz = torch.sigmoid(lf).detach().cpu().numpy()

        all_t.append(t.numpy())
        all_e.append(e.numpy())
        all_r.append(risks)
        all_h.append(haz)

    if not all_t:
        return {"c_index": 0.0, "ibs": 0.0, "auc_mean": float("nan"), "nb_mean": float("nan"), "nll": float("nan")}

    T = np.concatenate(all_t)
    E = np.concatenate(all_e)
    R = np.concatenate(all_r)
    HZ = np.concatenate(all_h, axis=0)

    c = H.concordance_index(T, E, R)
    ibs = H.integrated_brier_score(T, E, HZ, float(time_bin_width_days))
    auc_by, auc_mean = H.time_dependent_auc_surv(T, E, HZ, eval_times_days, float(time_bin_width_days))
    nb_by, nb_mean = H.decision_curve_analysis_surv(T, E, HZ, eval_times_days, dca_thresholds, float(time_bin_width_days))

    nll_mean = float(nll_sum / max(1, nll_n))

    out = {"c_index": float(c), "ibs": float(ibs), "auc_mean": float(auc_mean), "nb_mean": float(nb_mean), "nll": nll_mean}
    for tt, v in auc_by.items():
        out[f"auc_{int(round(tt))}d"] = float(v)
    for (tt, thr), v in nb_by.items():
        out[f"nb_{int(round(tt))}d_thr{str(thr).replace('.','p')}"] = float(v)
    return out


@torch.no_grad()
def predict_risk_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    endpoint: str,
    risk_horizon_days: float,
    autocast_ctx,
) -> Dict[str, float]:
    model.eval()
    mm = model.module if isinstance(model, nn.DataParallel) else model
    out: Dict[str, float] = {}

    for batch in loader:
        payload = _unpack_surv_batch(batch)
        x = payload["x"].to(device, non_blocking=True)
        clin = payload["clin"].to(device) if (payload["clin"] is not None and payload["clin"].numel() > 0) else None
        rad = payload["rad"].to(device) if (payload["rad"] is not None and payload["rad"].numel() > 0) else None
        with autocast_ctx():
            logits = model(x, clin, rad, teacher_force_alpha=0.0, return_gate=False)
            logits = _extract_endpoint_logits(logits, endpoint)
        risks = mm.hazards_to_risk(logits, horizon_days=float(risk_horizon_days)).detach().cpu().numpy()
        for i, p in enumerate(payload["pid"]):
            out[str(p)] = float(risks[i])
    return out


def save_risk_dict_csv(
    risk_dict: Dict[str, float],
    out_path: Path,
    id_col: str,
    endpoint: str = "",
    risk_horizon_days: float = float("nan"),
):
    save_endpoint_risk_dict_csv(
        risk_dict,
        out_path,
        id_col=id_col,
        endpoint=(str(endpoint).upper() if endpoint else "UNKNOWN"),
        risk_horizon_days=float(risk_horizon_days),
    )


def save_endpoint_risk_dict_csv(
    risk_dict: Dict[str, float],
    out_path: Path,
    *,
    id_col: str,
    endpoint: str,
    risk_horizon_days: float,
):
    rows = [
        {
            id_col: pid,
            "risk_score": float(score),
            "risk_endpoint": str(endpoint).upper(),
            "risk_horizon_days": float(risk_horizon_days),
        }
        for pid, score in risk_dict.items()
    ]
    pd.DataFrame(rows, columns=[id_col, "risk_score", "risk_endpoint", "risk_horizon_days"]).to_csv(out_path, index=False)
    print(f"[RISK] wrote {len(rows)} rows -> {out_path}")


# =============================================================================
# Training
# =============================================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scaler,
    device: torch.device,
    *,
    fold: int,
    epoch: int,
    primary_endpoint: str,
    num_time_bins: int,
    time_bin_width_days: float,
    primary_surv_loss_weight: float,
    aux_surv_loss_weight: float,
    hazard_smooth_lambda: float,
    logit_l2_lambda: float,
    gate_entropy_lambda: float,
    gate_loadbal_lambda: float,
    teacher_force_alpha: float,
    loc_loss_pt_lambda: float,
    loc_loss_ln_lambda: float,
    loc_presence_lambda: float,
    loc_bce_weight: float,
    loc_dice_weight: float,
    log_every_batches: int,
    ema: Optional[H.EMAWeights],
    autocast_ctx,
) -> Dict[str, float]:
    model.train()
    mm = model.module if isinstance(model, nn.DataParallel) else model
    request_aux = str(getattr(mm, "image_encoder_mode", "")).strip().lower() == "contour_aware"

    stats_sum: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_surv_total": 0.0,
        "loss_surv_primary": 0.0,
        "loss_surv_aux": 0.0,
        "loss_loc_pt": 0.0,
        "loss_loc_ln": 0.0,
        "loss_loc_presence": 0.0,
        "loss_gate_entropy": 0.0,
        "loss_gate_loadbal": 0.0,
        "loss_hazard_smooth": 0.0,
        "loss_logit_l2": 0.0,
    }
    for endpoint in SURVIVAL_ENDPOINTS:
        stats_sum[f"surv_{endpoint.lower()}_nll"] = 0.0
        stats_sum[f"surv_{endpoint.lower()}_count"] = 0.0
    n_batches = 0
    try:
        planned_batches = int(len(loader))
    except Exception:
        planned_batches = -1

    _log(
        f"[fold {fold:02d}] epoch {epoch:03d} starting"
        + (f" | batches={planned_batches}" if planned_batches > 0 else "")
        + f" | teacher_force={teacher_force_alpha:.2f}"
    )

    for batch_idx, batch in enumerate(loader, start=1):
        payload = _unpack_surv_batch(batch)
        x = payload["x"].to(device, non_blocking=True)
        t_all = payload["t_all"].to(device, non_blocking=True)
        e_all = payload["e_all"].to(device, non_blocking=True)
        clin = payload["clin"].to(device) if (payload["clin"] is not None and payload["clin"].numel() > 0) else None
        rad = payload["rad"].to(device) if (payload["rad"] is not None and payload["rad"].numel() > 0) else None
        mask_pt = _to_optional_device_tensor(payload["mask_pt"], device)
        mask_ln = _to_optional_device_tensor(payload["mask_ln"], device)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            if request_aux:
                logits, gate, pres, aux = model(
                    x,
                    clin,
                    rad,
                    mask_pt=mask_pt,
                    mask_ln=mask_ln,
                    teacher_force_alpha=float(teacher_force_alpha),
                    return_gate=True,
                    return_aux=True,
                )
            else:
                logits, gate, pres = model(
                    x,
                    clin,
                    rad,
                    mask_pt=mask_pt,
                    mask_ln=mask_ln,
                    teacher_force_alpha=float(teacher_force_alpha),
                    return_gate=True,
                    return_aux=False,
                )
                aux = None
            primary_logits = _extract_endpoint_logits(logits, primary_endpoint).float()
            endpoint_losses, endpoint_counts = _endpoint_survival_losses(
                logits,
                t_all=t_all,
                e_all=e_all,
                time_bin_width_days=float(time_bin_width_days),
                num_time_bins=int(num_time_bins),
            )

            loss_surv = _weighted_multitask_mean(
                endpoint_losses,
                endpoint_counts,
                primary_endpoint=primary_endpoint,
                primary_weight=float(primary_surv_loss_weight),
                aux_weight=float(aux_surv_loss_weight),
                ref_tensor=primary_logits,
            )
            loss = loss_surv
            primary_key = str(primary_endpoint).strip().upper()
            primary_surv_unweighted = endpoint_losses.get(primary_key, primary_logits.new_tensor(0.0))
            aux_endpoint_losses = {endpoint: loss_ep for endpoint, loss_ep in endpoint_losses.items() if endpoint != primary_key}
            aux_endpoint_counts = {endpoint: endpoint_counts.get(endpoint, 0) for endpoint in aux_endpoint_losses}
            aux_surv_unweighted = _weighted_multitask_mean(
                aux_endpoint_losses,
                aux_endpoint_counts,
                primary_endpoint=primary_key,
                primary_weight=1.0,
                aux_weight=1.0,
                ref_tensor=primary_logits,
            )
            loss_hazard_smooth = primary_logits.new_tensor(0.0)
            loss_logit_l2 = primary_logits.new_tensor(0.0)
            loss_gate_entropy = primary_logits.new_tensor(0.0)
            loss_gate_loadbal = primary_logits.new_tensor(0.0)
            loss_loc_pt = primary_logits.new_tensor(0.0)
            loss_loc_ln = primary_logits.new_tensor(0.0)
            loss_loc_presence = primary_logits.new_tensor(0.0)
            if hazard_smooth_lambda > 0:
                hazard_terms = {
                    endpoint: H.hazard_smoothness_penalty(_extract_endpoint_logits(logits, endpoint).float())
                    for endpoint in endpoint_losses
                }
                loss_hazard_smooth = float(hazard_smooth_lambda) * _weighted_multitask_mean(
                    hazard_terms,
                    endpoint_counts,
                    primary_endpoint=primary_key,
                    primary_weight=float(primary_surv_loss_weight),
                    aux_weight=float(aux_surv_loss_weight),
                    ref_tensor=primary_logits,
                )
                loss = loss + loss_hazard_smooth
            if logit_l2_lambda > 0:
                l2_terms = {
                    endpoint: (_extract_endpoint_logits(logits, endpoint).float() ** 2).mean()
                    for endpoint in endpoint_losses
                }
                loss_logit_l2 = float(logit_l2_lambda) * _weighted_multitask_mean(
                    l2_terms,
                    endpoint_counts,
                    primary_endpoint=primary_key,
                    primary_weight=float(primary_surv_loss_weight),
                    aux_weight=float(aux_surv_loss_weight),
                    ref_tensor=primary_logits,
                )
                loss = loss + loss_logit_l2
            if gate_entropy_lambda > 0:
                loss_gate_entropy = float(gate_entropy_lambda) * gate_entropy_penalty_presence(gate, pres)
                loss = loss + loss_gate_entropy
            if gate_loadbal_lambda > 0:
                loss_gate_loadbal = float(gate_loadbal_lambda) * gate_load_balance_penalty_presence(gate, pres)
                loss = loss + loss_gate_loadbal

            if aux is not None and mask_pt is not None and mask_ln is not None:
                loc_pt_target = _resize_mask_target(mask_pt, tuple(int(x) for x in aux["loc_pt_logits"].shape[2:]))
                loc_ln_target = _resize_mask_target(mask_ln, tuple(int(x) for x in aux["loc_ln_logits"].shape[2:]))
                if float(loc_loss_pt_lambda) > 0.0:
                    loss_loc_pt = float(loc_loss_pt_lambda) * _localization_loss_from_logits(
                        aux["loc_pt_logits"].float(),
                        loc_pt_target.float(),
                        bce_weight=float(loc_bce_weight),
                        dice_weight=float(loc_dice_weight),
                    )
                    loss = loss + loss_loc_pt
                if float(loc_loss_ln_lambda) > 0.0:
                    loss_loc_ln = float(loc_loss_ln_lambda) * _localization_loss_from_logits(
                        aux["loc_ln_logits"].float(),
                        loc_ln_target.float(),
                        bce_weight=float(loc_bce_weight),
                        dice_weight=float(loc_dice_weight),
                    )
                    loss = loss + loss_loc_ln
                if float(loc_presence_lambda) > 0.0 and ("pt_presence_logits" in aux) and ("ln_presence_logits" in aux):
                    pt_present_target = (mask_pt.flatten(1).sum(dim=1) > 0.0).float()
                    ln_present_target = (mask_ln.flatten(1).sum(dim=1) > 0.0).float()
                    pres_loss = F.binary_cross_entropy_with_logits(aux["pt_presence_logits"].float(), pt_present_target)
                    pres_loss = pres_loss + F.binary_cross_entropy_with_logits(aux["ln_presence_logits"].float(), ln_present_target)
                    loss_loc_presence = float(loc_presence_lambda) * pres_loss
                    loss = loss + loss_loc_presence

        if not torch.isfinite(loss).item():
            pid_preview = [str(x) for x in list(payload["pid"])[:8]]
            raise RuntimeError(
                "[TRAIN][NONFINITE] non-finite loss encountered "
                f"primary={primary_key} teacher_force={teacher_force_alpha:.3f} "
                f"pids={pid_preview} "
                f"loss_total={float(loss.detach().cpu().item())} "
                f"loss_surv={float(loss_surv.detach().cpu().item())} "
                f"loss_surv_primary={float(primary_surv_unweighted.detach().cpu().item())} "
                f"loss_surv_aux={float(aux_surv_unweighted.detach().cpu().item()) if torch.isfinite(aux_surv_unweighted).item() else float('nan')} "
                f"loss_loc_pt={float(loss_loc_pt.detach().cpu().item())} "
                f"loss_loc_ln={float(loss_loc_ln.detach().cpu().item())} "
                f"loss_loc_presence={float(loss_loc_presence.detach().cpu().item())} "
                f"loss_gate_entropy={float(loss_gate_entropy.detach().cpu().item())} "
                f"loss_gate_loadbal={float(loss_gate_loadbal.detach().cpu().item())} "
                f"loss_hazard_smooth={float(loss_hazard_smooth.detach().cpu().item())} "
                f"loss_logit_l2={float(loss_logit_l2.detach().cpu().item())} "
                f"endpoint_counts={endpoint_counts}"
            )

        if device.type == "cuda" and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if ema is not None:
            ema.update(mm)

        n_batches += 1
        stats_sum["loss_total"] += float(loss.detach().cpu().item())
        stats_sum["loss_surv_total"] += float(loss_surv.detach().cpu().item())
        stats_sum["loss_surv_primary"] += float(primary_surv_unweighted.detach().cpu().item())
        stats_sum["loss_surv_aux"] += float(aux_surv_unweighted.detach().cpu().item()) if aux_endpoint_losses else 0.0
        stats_sum["loss_loc_pt"] += float(loss_loc_pt.detach().cpu().item())
        stats_sum["loss_loc_ln"] += float(loss_loc_ln.detach().cpu().item())
        stats_sum["loss_loc_presence"] += float(loss_loc_presence.detach().cpu().item())
        stats_sum["loss_gate_entropy"] += float(loss_gate_entropy.detach().cpu().item())
        stats_sum["loss_gate_loadbal"] += float(loss_gate_loadbal.detach().cpu().item())
        stats_sum["loss_hazard_smooth"] += float(loss_hazard_smooth.detach().cpu().item())
        stats_sum["loss_logit_l2"] += float(loss_logit_l2.detach().cpu().item())
        for endpoint in SURVIVAL_ENDPOINTS:
            if endpoint in endpoint_losses:
                stats_sum[f"surv_{endpoint.lower()}_nll"] += float(endpoint_losses[endpoint].detach().cpu().item())
            stats_sum[f"surv_{endpoint.lower()}_count"] += float(endpoint_counts.get(endpoint, 0))

        should_log_batch = (
            batch_idx == 1
            or (int(log_every_batches) > 0 and batch_idx % int(log_every_batches) == 0)
            or (planned_batches > 0 and batch_idx == planned_batches)
        )
        if should_log_batch:
            batch_total = planned_batches if planned_batches > 0 else "?"
            _log(
                f"[fold {fold:02d}] epoch {epoch:03d} batch {batch_idx}/{batch_total} "
                f"loss={float(loss.detach().cpu().item()):.4f} "
                f"surv={float(loss_surv.detach().cpu().item()):.4f} "
                f"loc_pt={float(loss_loc_pt.detach().cpu().item()):.4f} "
                f"loc_ln={float(loss_loc_ln.detach().cpu().item()):.4f}"
            )

    if n_batches == 0:
        return {k: float("nan") for k in stats_sum}

    stats_mean: Dict[str, float] = {}
    for key, value in stats_sum.items():
        if key.endswith("_count"):
            stats_mean[key] = float(value / n_batches)
        else:
            stats_mean[key] = float(value / n_batches)
    return stats_mean


# =============================================================================
# Checkpointing
# =============================================================================
def save_checkpoint(
    path: Path,
    epoch: int,
    num_time_bins: int,
    model: nn.Module,
    args: argparse.Namespace,
    optimizer,
    scaler,
    scheduler,
    report_metric: float,
    ema: Optional[H.EMAWeights],
    swa: Optional[H.SWAWeights],
):
    state = {
        "epoch": int(epoch),
        "num_time_bins": int(num_time_bins),
        "model_state": (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": (scaler.state_dict() if scaler is not None else None),
        "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
        "report_metric": float(report_metric) if report_metric is not None else float("nan"),
        "ema": (ema.state_dict() if ema is not None else None),
        "swa": (swa.state_dict() if swa is not None else None),
        "args": dict(vars(args)),
    }
    torch.save(state, path)


def checkpoint_report_metric(path: Path) -> float:
    if not path.is_file():
        return float("-inf")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        metric = ck.get("report_metric", float("-inf"))
        metric = float(metric)
        return metric if np.isfinite(metric) else float("-inf")
    except Exception:
        return float("-inf")


def load_model_state_only(path: Path, model: nn.Module) -> bool:
    if not path.is_file():
        return False
    ck = torch.load(path, map_location="cpu", weights_only=False)
    in_sd = ck.get("model_state", {})
    if not isinstance(in_sd, dict):
        return False

    target_sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    filtered = {}
    for k, v in in_sd.items():
        if k in target_sd and tuple(target_sd[k].shape) == tuple(v.shape):
            filtered[k] = v

    target_head_keys = {k for k in target_sd if k.startswith("surv_heads.")}
    filtered_head_keys = {k for k in filtered if k.startswith("surv_heads.")}
    if target_head_keys and filtered_head_keys != target_head_keys:
        missing = sorted(target_head_keys - filtered_head_keys)
        print(f"[CKPT][WARN] refusing partial survival-head restore from {path}; missing head keys like {missing[:5]}")
        return False

    if not filtered:
        return False

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(filtered, strict=False)
    return True


def load_checkpoint(
    path: Path,
    num_time_bins: int,
    model: nn.Module,
    optimizer,
    scaler,
    scheduler,
    ema: Optional[H.EMAWeights],
    swa: Optional[H.SWAWeights],
) -> Tuple[int, bool]:
    """
    Returns: (next_epoch, fully_restored)
    fully_restored=False means optimizer/scheduler may not match, or many keys mismatched.
    """
    if not path.is_file():
        return 1, False
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if int(ck.get("num_time_bins", -1)) != int(num_time_bins):
        raise RuntimeError(f"Checkpoint num_time_bins mismatch: ckpt={ck.get('num_time_bins')} current={num_time_bins}")

    in_sd = ck.get("model_state", {})
    target_sd = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

    filtered = {}
    mismatched = 0
    for k, v in in_sd.items():
        if k in target_sd and tuple(target_sd[k].shape) == tuple(v.shape):
            filtered[k] = v
        else:
            mismatched += 1

    target_head_keys = {k for k in target_sd if k.startswith("surv_heads.")}
    filtered_head_keys = {k for k in filtered if k.startswith("surv_heads.")}
    if target_head_keys and filtered_head_keys != target_head_keys:
        missing = sorted(target_head_keys - filtered_head_keys)
        raise RuntimeError(
            f"[CKPT] checkpoint {path} is missing current multitask survival head keys; "
            f"examples={missing[:5]}"
        )

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(filtered, strict=False)
    else:
        model.load_state_dict(filtered, strict=False)

    fully_restored = (mismatched == 0)

    try:
        optimizer.load_state_dict(ck.get("optimizer_state", {}))
    except Exception as ex:
        fully_restored = False
        print(f"[CKPT][WARN] optimizer restore failed: {ex}")

    if scaler is not None and ck.get("scaler_state") is not None:
        try:
            scaler.load_state_dict(ck["scaler_state"])
        except Exception as ex:
            fully_restored = False
            print(f"[CKPT][WARN] scaler restore failed: {ex}")

    if scheduler is not None and ck.get("scheduler_state") is not None:
        try:
            scheduler.load_state_dict(ck["scheduler_state"])
        except Exception as ex:
            fully_restored = False
            print(f"[CKPT][WARN] scheduler restore failed: {ex}")

    mm = model.module if isinstance(model, nn.DataParallel) else model
    if ema is not None and ck.get("ema") is not None:
        try:
            ema.load_state_dict(ck["ema"], model=mm)
        except Exception as ex:
            fully_restored = False
            print(f"[CKPT][WARN] EMA restore failed: {ex}")
    if swa is not None and ck.get("swa") is not None:
        try:
            swa.load_state_dict(ck["swa"], model=mm)
        except Exception as ex:
            fully_restored = False
            print(f"[CKPT][WARN] SWA restore failed: {ex}")

    if mismatched > 0:
        print(f"[CKPT][WARN] {mismatched} mismatched key(s); model warm-started with filtered keys only.")
        fully_restored = False

    last_epoch = int(ck.get("epoch", 0))
    return last_epoch + 1, fully_restored


# =============================================================================
# Fold runner helpers
# =============================================================================
def select_df_by_ids(meta: pd.DataFrame, ids: List[str], id_col: str, strict: bool, tag: str) -> pd.DataFrame:
    ids = [str(x) for x in ids]
    have = set(meta[id_col].astype(str).tolist())
    missing = [x for x in ids if x not in have]
    if missing:
        msg = f"[SPLIT][{tag}] {len(missing)} id(s) missing in meta. First: {missing[:10]}"
        if strict:
            raise RuntimeError(msg)
        print("[WARN]", msg, "-> dropping missing")
        ids = [x for x in ids if x in have]
    return meta[meta[id_col].astype(str).isin(set(ids))].copy().reset_index(drop=True)


def _candidate_contour_warmstart_names(args) -> List[str]:
    names: List[str] = []
    for value in [
        getattr(args, "contour_warmstart_name", "best.pt"),
        getattr(args, "shared_seg_pretrain_name", "best.pt"),
        "best.pt",
        "seg_best.pt",
    ]:
        name = str(value or "").strip()
        if name and name not in names:
            names.append(name)
    return names


def _resolve_existing_contour_warmstart_ckpt_for_cfg(args) -> str:
    ckpt = getattr(args, "contour_warmstart_ckpt", "") or getattr(args, "shared_seg_pretrain_ckpt", "") or ""
    if ckpt and os.path.isfile(ckpt):
        return ckpt

    d = getattr(args, "contour_warmstart_dir", "") or getattr(args, "shared_seg_pretrain_dir", "") or ""
    if d:
        for name in _candidate_contour_warmstart_names(args):
            p_all = os.path.join(d, "all", name)
            if os.path.isfile(p_all):
                return p_all
            for f in range(int(getattr(args, "cv_folds", 4))):
                p = os.path.join(d, f"fold_{f:02d}", name)
                if os.path.isfile(p):
                    return p
    return ""


def _resolve_contour_warmstart_ckpt_for_fold(args, fold: int) -> str:
    ckpt = getattr(args, "contour_warmstart_ckpt", "") or getattr(args, "shared_seg_pretrain_ckpt", "") or ""
    if ckpt and os.path.isfile(ckpt):
        return ckpt

    d = getattr(args, "contour_warmstart_dir", "") or getattr(args, "shared_seg_pretrain_dir", "") or ""
    if d:
        for name in _candidate_contour_warmstart_names(args):
            p_all = os.path.join(d, "all", name)
            if os.path.isfile(p_all):
                return p_all
            p_fold = os.path.join(d, f"fold_{int(fold):02d}", name)
            if os.path.isfile(p_fold):
                return p_fold
    return ckpt


def _image_encoder_mode(args) -> str:
    mode = str(getattr(args, "image_encoder_mode", "contour_aware")).strip().lower()
    if mode != "contour_aware":
        raise ValueError(f"Unsupported image_encoder_mode for compact package: {mode}")
    return mode


def _iter_image_encoder_backbones(img_backbone):
    if hasattr(img_backbone, "iter_encoder_backbones"):
        return list(img_backbone.iter_encoder_backbones())
    if hasattr(img_backbone, "backbone_shared"):
        return [("backbone_shared", img_backbone.backbone_shared)]
    return []


def _selected_image_encoder_backbones(img_backbone, scope: str):
    pairs = _iter_image_encoder_backbones(img_backbone)
    return pairs


def _list_trainable_param_names(mod: nn.Module, prefix: str = "", limit: int = 80) -> List[str]:
    out = []
    for n, p in mod.named_parameters():
        if p.requires_grad:
            out.append(f"{prefix}{n}  shape={tuple(p.shape)}")
            if len(out) >= limit:
                out.append("... (truncated)")
                break
    return out


def _assert_backbone_trainables(backbone: nn.Module, allow_lora: bool):
    """
    Strong sanity check: ensure trainable params under backbone are only LoRA.
    This is conservative but prevents accidental full fine-tuning.
    """
    for n, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        ln = n.lower()
        is_lora = (".lora_a." in ln) or (".lora_b." in ln)
        if is_lora and allow_lora:
            continue
        raise RuntimeError(
            f"[POLICY] Unexpected trainable backbone param: {n} shape={tuple(p.shape)} "
            f"(allow_lora={allow_lora})"
        )


def apply_lora_to_image_encoder(base_model: SwinUNETRTokenMoEDiscrete, args) -> int:
    if not bool(args.use_lora):
        return 0

    scope = str(args.lora_scope).lower().strip()
    targets = list(args.lora_targets or [])
    r = int(args.lora_r)
    alpha = float(args.lora_alpha)
    drop = float(args.lora_dropout)

    selected = _selected_image_encoder_backbones(base_model.img_backbone, scope)
    if not selected:
        raise RuntimeError(f"[LoRA] no encoder backbones selected for scope={scope}")

    n_total = 0
    summaries = []
    for prefix, bb in selected:
        n_added = inject_lora_into_module(bb, target_keywords=targets, r=r, alpha=alpha, dropout=drop, verbose=True)
        n_total += n_added
        mark_only_lora_trainable(bb)
        a_cnt, t_cnt = count_trainable(bb)
        summaries.append(f"{prefix} trainable {t_cnt:,}/{a_cnt:,}")

    print(f"[LoRA] {' | '.join(summaries)} | injected_total={n_total}")

    if int(args.lora_min_replacements) > 0 and n_total < int(args.lora_min_replacements):
        raise RuntimeError(f"[LoRA] injected_total={n_total} < --lora_min_replacements={args.lora_min_replacements}. Check naming/targets.")
    if n_total == 0:
        raise RuntimeError("[LoRA] --use_lora requested but injected_total=0. Check encoder mode / naming / targets.")

    return n_total


def _reset_metrics_log_if_needed(metrics_csv: Path, start_epoch: int, fully_restored: bool, args):
    """
    Avoid mixed histories:
    - If resuming properly (start_epoch>1 and fully_restored), keep appending.
    - Otherwise, archive existing metrics.csv if present and start from fresh.
    """
    if not metrics_csv.is_file():
        return
    if start_epoch > 1 and fully_restored:
        return
    # archive
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    dst = metrics_csv.with_name(f"metrics_archived_{ts}.csv")
    metrics_csv.rename(dst)
    print(f"[LOG] archived old metrics -> {dst}")


# =============================================================================
# run_one_fold
# =============================================================================
def run_one_fold(
    fold: int,
    args,
    meta: pd.DataFrame,
    split: Dict[str, List[str]],
    *,
    device: torch.device,
    scaler,
    autocast_ctx,
    out_root: Path,
) -> Dict[str, Any]:
    set_seed(args.seed + 100 * int(fold))
    _assert_split_disjoint(int(fold), split)
    data_root = str(Path(args.meta_csv).resolve().parent)

    fold_dir = out_root / f"fold_{fold:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = fold_dir / "metrics.csv"
    ckpt_last = fold_dir / "last.pt"
    ckpt_best = fold_dir / "best.pt"
    best_report_metric = checkpoint_report_metric(ckpt_best)

    tr_df = select_df_by_ids(meta, split["train"], args.id_col, args.strict_splits, f"fold{fold:02d}/train")
    va_df = select_df_by_ids(meta, split["val"],   args.id_col, args.strict_splits, f"fold{fold:02d}/val")
    te_df = select_df_by_ids(meta, split["test"],  args.id_col, args.strict_splits, f"fold{fold:02d}/test")

    if args.debug_max_train > 0:
        tr_df = tr_df.iloc[: int(args.debug_max_train)].reset_index(drop=True)
    if args.debug_max_val > 0:
        va_df = va_df.iloc[: int(args.debug_max_val)].reset_index(drop=True)
    if args.debug_max_test > 0:
        te_df = te_df.iloc[: int(args.debug_max_test)].reset_index(drop=True)

    print(f"[fold {fold:02d}] train={len(tr_df)} val={len(va_df)} test={len(te_df)}")

    clin_enc = ClinicalEncoder.fit(tr_df, args.clinical_cols)
    clin_dim = clin_enc.output_dim

    rad_enc = None
    rad_dim = 0
    if args.use_radiomics:
        all_ids = pd.concat([tr_df[args.id_col], va_df[args.id_col], te_df[args.id_col]]).astype(str).unique().tolist()
        train_ids = tr_df[args.id_col].astype(str).tolist()
        rad_enc = RadiomicsEncoder.fit(train_ids, all_ids, args.radiomics_root, args.radiomics_pca_total_components, args.seed)
        rad_dim = rad_enc.output_dim
        print(f"[RAD] radiomics_dim={rad_dim}")

    expected_dhw = tuple(int(x) for x in args.img_size)
    mode = _image_encoder_mode(args)
    primary_valid_train = _valid_survival_mask_frame(tr_df, args.time_col, args.event_col)
    tr_eval_df = tr_df.loc[primary_valid_train].copy().reset_index(drop=True)
    if tr_eval_df.empty:
        raise RuntimeError(f"[fold {fold:02d}] no primary-endpoint-labeled training rows remain for evaluation")
    if not bool(_valid_survival_mask_frame(va_df, args.time_col, args.event_col).all()):
        raise RuntimeError(f"[fold {fold:02d}] validation split contains rows without valid primary endpoint labels")
    if not bool(_valid_survival_mask_frame(te_df, args.time_col, args.event_col).all()):
        raise RuntimeError(f"[fold {fold:02d}] test split contains rows without valid primary endpoint labels")

    fold_num_time_bins, fold_max_time = compute_multitask_num_time_bins(tr_df, args)
    print(f"[TIME][fold {fold:02d}] train num_time_bins={fold_num_time_bins} train_max_time={fold_max_time:.1f}d")

    tr_ds = PreprocessedContourAwareDataset(
        tr_df,
        id_col=args.id_col, time_col=args.time_col, event_col=args.event_col,
        multi_time_cols=MULTITASK_TIME_COLS, multi_event_cols=MULTITASK_EVENT_COLS,
        ct_col=args.ct_col, mask_pt_col=args.mask_pt_col, mask_ln_col=args.mask_ln_col,
        mode="train",
        clinical_encoder=clin_enc, radiomics_encoder=rad_enc,
        use_radiomics=args.use_radiomics,
        strict_files=args.strict_files,
        expected_dhw=expected_dhw,
        data_root=data_root,
    )
    tr_eval_ds = PreprocessedContourAwareDataset(
        tr_eval_df,
        id_col=args.id_col, time_col=args.time_col, event_col=args.event_col,
        multi_time_cols=MULTITASK_TIME_COLS, multi_event_cols=MULTITASK_EVENT_COLS,
        ct_col=args.ct_col, mask_pt_col=args.mask_pt_col, mask_ln_col=args.mask_ln_col,
        mode="eval",
        clinical_encoder=clin_enc, radiomics_encoder=rad_enc,
        use_radiomics=args.use_radiomics,
        strict_files=args.strict_files,
        expected_dhw=expected_dhw,
        data_root=data_root,
    )
    va_ds = PreprocessedContourAwareDataset(
        va_df,
        id_col=args.id_col, time_col=args.time_col, event_col=args.event_col,
        multi_time_cols=MULTITASK_TIME_COLS, multi_event_cols=MULTITASK_EVENT_COLS,
        ct_col=args.ct_col, mask_pt_col=args.mask_pt_col, mask_ln_col=args.mask_ln_col,
        mode="eval",
        clinical_encoder=clin_enc, radiomics_encoder=rad_enc,
        use_radiomics=args.use_radiomics,
        strict_files=args.strict_files,
        expected_dhw=expected_dhw,
        data_root=data_root,
    )
    te_ds = PreprocessedContourAwareDataset(
        te_df,
        id_col=args.id_col, time_col=args.time_col, event_col=args.event_col,
        multi_time_cols=MULTITASK_TIME_COLS, multi_event_cols=MULTITASK_EVENT_COLS,
        ct_col=args.ct_col, mask_pt_col=args.mask_pt_col, mask_ln_col=args.mask_ln_col,
        mode="eval",
        clinical_encoder=clin_enc, radiomics_encoder=rad_enc,
        use_radiomics=args.use_radiomics,
        strict_files=args.strict_files,
        expected_dhw=expected_dhw,
        data_root=data_root,
    )

    g = torch.Generator()
    g.manual_seed(int(args.seed + 777 + fold))

    tr_loader = DataLoader(tr_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.workers),
                           pin_memory=(device.type == "cuda"), drop_last=False, persistent_workers=(int(args.workers) > 0),
                           worker_init_fn=seed_worker, generator=g)
    tr_eval_loader = DataLoader(tr_eval_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.workers),
                                pin_memory=(device.type == "cuda"), drop_last=False, persistent_workers=(int(args.workers) > 0),
                                worker_init_fn=seed_worker)
    va_loader = DataLoader(va_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.workers),
                           pin_memory=(device.type == "cuda"), drop_last=False, persistent_workers=(int(args.workers) > 0),
                           worker_init_fn=seed_worker)
    te_loader = DataLoader(te_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.workers),
                           pin_memory=(device.type == "cuda"), drop_last=False, persistent_workers=(int(args.workers) > 0),
                           worker_init_fn=seed_worker)

    backbone_cfg = dict(
        img_size=tuple(args.img_size),
        feature_size=int(args.feature_size),
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        drop_rate=float(args.drop_rate),
        attn_drop_rate=float(args.attn_drop_rate),
        dropout_path_rate=float(args.dropout_path_rate),
        normalize=True,
        use_checkpoint=bool(args.use_checkpoint),
        image_encoder_mode=mode,

        token_dim=int(args.img_token_dim),
        token_mlp_dropout=float(args.token_mlp_dropout),
        token_mlp_hidden_dim=int(args.token_mlp_hidden_dim),

        attn_mask_bias=float(args.attn_mask_bias),
        use_multiscale=bool(args.use_multiscale),
        mask_interp=str(args.mask_interp),
        min_roi_frac=float(args.min_roi_frac),
        min_roi_voxels_deep=int(args.min_roi_voxels_deep),
        token_dropout=float(args.token_dropout),

        pt_shell_radius=int(args.pt_shell_radius),
        ln_shell_radius=int(args.ln_shell_radius),
        shell_body_from_ct=bool(args.shell_body_from_ct),
        body_ct_thr=str(args.body_ct_thr),
        body_ct_thr_hu=float(args.body_ct_thr_hu),
        body_close_r=int(args.body_close_r),
        body_max_frac=float(args.body_max_frac),
        strict_swinvit_layout=bool(args.strict_swinvit_layout),
        debug_swinvit_layout=bool(args.debug_swinvit_layout),
    )
    backbone_cfg.update(dict(
        force_presence_from_raw_masks=False,
        raw_mask_threshold=0.5,
        fallback_peri_to_intra=True,
    ))

    base_model = SwinUNETRTokenMoEDiscrete(
        num_time_bins=int(fold_num_time_bins),
        time_bin_width_days=float(args.time_bin_width_days),
        fused_dim=int(args.fused_dim),
        backbone_cfg=backbone_cfg,
        clinical_dim=int(clin_dim),
        radiomics_dim=int(rad_dim),
        expert_dropout_p=float(args.expert_dropout_p),
        proj_dropout_p=float(args.proj_dropout_p),
        attn_dropout_p=float(args.attn_dropout_p),
        gate_dropout_p=float(args.gate_dropout_p),
        surv_dropout_p=float(args.surv_dropout_p),
        clinical_noise_std=float(args.clinical_noise_std),
        radiomics_noise_std=float(args.radiomics_noise_std),
        modality_dropout_clin_p=float(args.modality_dropout_clin_p),
        modality_dropout_rad_p=float(args.modality_dropout_rad_p),
        img_proj_hidden_dim=int(args.img_proj_hidden_dim),
        img_tok_ffn_hidden_dim=int(args.img_tok_ffn_hidden_dim),
        img_post_hidden_dim=int(args.img_post_hidden_dim),
        img_attn_heads=int(args.img_attn_heads),
        gate_hidden_dim=int(args.gate_hidden_dim),
        rad_hidden_dim=int(args.rad_hidden_dim),
        rad_proj_dropout_p=float(args.rad_proj_dropout_p),
        nan_guard=bool(args.nan_guard),
    ).to(device)

    injected = 0

    contour_ckpt = _resolve_contour_warmstart_ckpt_for_fold(args, fold=int(fold))
    if contour_ckpt and os.path.isfile(contour_ckpt):
        print(f"[SWIN][CONTOUR][fold {fold:02d}] loading contour-aware warm-start: {contour_ckpt}")
        load_swinunetr_pretrained(base_model.img_backbone.backbone_shared, contour_ckpt, verbose=True, allow_inflate_patch_embed=True)
    else:
        print(f"[SWIN][CONTOUR][fold {fold:02d}] contour-aware warm-start not found/disabled: {contour_ckpt}")

    if bool(args.use_lora):
        freeze_all_params(base_model.img_backbone.backbone_shared)
        print("[ENC] policy: contour-aware backbone frozen + LoRA trainable")
        injected = apply_lora_to_image_encoder(base_model, args)
        for _, bb in _iter_image_encoder_backbones(base_model.img_backbone):
            _assert_backbone_trainables(bb, allow_lora=True)
    else:
        print("[ENC] policy: contour-aware backbone full fine-tune for survival")

    if args.print_trainable_backbone_params:
        for prefix, bb in _iter_image_encoder_backbones(base_model.img_backbone):
            print(f"[DBG] Trainable {prefix} params:")
            for s in _list_trainable_param_names(bb, prefix=f"{prefix}."):
                print("  ", s)

    materialize_lazy_modules(base_model, tr_loader, device, autocast_ctx)

    if args.data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(base_model)
    else:
        model = base_model

    mm = model.module if isinstance(model, nn.DataParallel) else model

    groups = make_param_groups(
        mm,
        lr_backbone=float(args.lr_backbone),
        lr_lora=float(args.lr_lora),
        lr_head=float(args.lr_head),
        wd_backbone=float(args.wd_backbone),
        wd_lora=float(args.wd_lora),
        wd_head=float(args.wd_head),
        wd_clin=float(args.wd_clin),
        wd_rad=float(args.wd_rad),
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=float(args.min_lr))

    ema = H.EMAWeights(mm, decay=float(args.ema_decay), track_trainable_only=True) if args.use_ema else None
    swa = H.SWAWeights(mm, track_trainable_only=True) if args.use_swa else None

    start_epoch = 1
    fully_restored = False
    if args.resume:
        start_epoch, fully_restored = load_checkpoint(ckpt_last, int(fold_num_time_bins), model, optimizer, scaler, scheduler, ema, swa)
        if start_epoch > 1:
            print(f"[fold {fold:02d}] resumed from epoch {start_epoch} (fully_restored={fully_restored})")

    _reset_metrics_log_if_needed(metrics_csv, start_epoch=start_epoch, fully_restored=fully_restored, args=args)

    print(f"[fold {fold:02d}] training epochs {start_epoch}..{args.epochs} (no early stop)")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        teacher_force_alpha = _teacher_force_alpha_for_epoch(epoch, args)
        train_stats = train_one_epoch(
            model, tr_loader, optimizer, scaler, device,
            fold=int(fold),
            epoch=int(epoch),
            primary_endpoint=str(args.endpoint),
            num_time_bins=int(fold_num_time_bins),
            time_bin_width_days=float(args.time_bin_width_days),
            primary_surv_loss_weight=float(args.primary_surv_loss_weight),
            aux_surv_loss_weight=float(args.aux_surv_loss_weight),
            hazard_smooth_lambda=float(args.hazard_smooth_lambda),
            logit_l2_lambda=float(args.logit_l2_lambda),
            gate_entropy_lambda=float(args.gate_entropy_lambda),
            gate_loadbal_lambda=float(args.gate_loadbal_lambda),
            teacher_force_alpha=float(teacher_force_alpha),
            loc_loss_pt_lambda=float(args.loc_loss_pt_lambda),
            loc_loss_ln_lambda=float(args.loc_loss_ln_lambda),
            loc_presence_lambda=float(args.loc_presence_lambda),
            loc_bce_weight=float(args.loc_bce_weight),
            loc_dice_weight=float(args.loc_dice_weight),
            log_every_batches=int(args.log_every_batches),
            ema=ema,
            autocast_ctx=autocast_ctx,
        )

        train_met = evaluate_model(
            model, tr_eval_loader, device,
            endpoint=str(args.endpoint),
            risk_horizon_days=float(args.risk_horizon_days),
            time_bin_width_days=float(args.time_bin_width_days),
            eval_times_days=args.auc_times_days,
            dca_thresholds=args.dca_thresholds,
            autocast_ctx=autocast_ctx,
        )
        val_met = evaluate_model(
            model, va_loader, device,
            endpoint=str(args.endpoint),
            risk_horizon_days=float(args.risk_horizon_days),
            time_bin_width_days=float(args.time_bin_width_days),
            eval_times_days=args.auc_times_days,
            dca_thresholds=args.dca_thresholds,
            autocast_ctx=autocast_ctx,
        )

        if swa is not None and epoch >= int(args.swa_start_epoch):
            if ((epoch - int(args.swa_start_epoch)) % int(args.swa_update_freq_epochs)) == 0:
                swa.update(mm)

        report_metric = float(val_met.get(args.report_metric, float("nan")))
        print(
            f"[fold {fold:02d}] epoch {epoch:03d} | "
            f"primary={str(args.endpoint).upper()} | "
            f"loss={train_stats.get('loss_total', float('nan')):.4f} "
            f"surv={train_stats.get('loss_surv_total', float('nan')):.4f} "
            f"prim={train_stats.get('loss_surv_primary', float('nan')):.4f} "
            f"aux={train_stats.get('loss_surv_aux', float('nan')):.4f} "
            f"os={train_stats.get('surv_os_nll', float('nan')):.4f} "
            f"dss={train_stats.get('surv_dss_nll', float('nan')):.4f} "
            f"dfs={train_stats.get('surv_dfs_nll', float('nan')):.4f} | "
            f"loc_pt={train_stats.get('loss_loc_pt', float('nan')):.4f} "
            f"loc_ln={train_stats.get('loss_loc_ln', float('nan')):.4f} "
            f"loc_pres={train_stats.get('loss_loc_presence', float('nan')):.4f} | "
            f"val_nll={val_met.get('nll', float('nan')):.4f} | "
            f"train_c={train_met.get('c_index', float('nan')):.3f} val_c={val_met.get('c_index', float('nan')):.3f} "
            f"val_{args.report_metric}={report_metric:.3f} "
            f"teacher_force={teacher_force_alpha:.2f}"
        )

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_stats.get("loss_total", float("nan"))),
            "train_surv_loss": float(train_stats.get("loss_surv_total", float("nan"))),
            "train_surv_primary_loss": float(train_stats.get("loss_surv_primary", float("nan"))),
            "train_surv_aux_loss": float(train_stats.get("loss_surv_aux", float("nan"))),
            "train_surv_os_nll": float(train_stats.get("surv_os_nll", float("nan"))),
            "train_surv_dss_nll": float(train_stats.get("surv_dss_nll", float("nan"))),
            "train_surv_dfs_nll": float(train_stats.get("surv_dfs_nll", float("nan"))),
            "train_surv_os_valid_per_batch": float(train_stats.get("surv_os_count", float("nan"))),
            "train_surv_dss_valid_per_batch": float(train_stats.get("surv_dss_count", float("nan"))),
            "train_surv_dfs_valid_per_batch": float(train_stats.get("surv_dfs_count", float("nan"))),
            "train_loc_pt_loss": float(train_stats.get("loss_loc_pt", float("nan"))),
            "train_loc_ln_loss": float(train_stats.get("loss_loc_ln", float("nan"))),
            "train_loc_presence_loss": float(train_stats.get("loss_loc_presence", float("nan"))),
            "train_gate_entropy_loss": float(train_stats.get("loss_gate_entropy", float("nan"))),
            "train_gate_loadbal_loss": float(train_stats.get("loss_gate_loadbal", float("nan"))),
            "train_hazard_smooth_loss": float(train_stats.get("loss_hazard_smooth", float("nan"))),
            "train_logit_l2_loss": float(train_stats.get("loss_logit_l2", float("nan"))),
            "train_c_index": float(train_met.get("c_index", float("nan"))),
            "val_c_index": float(val_met.get("c_index", float("nan"))),
            "train_ibs": float(train_met.get("ibs", float("nan"))),
            "val_ibs": float(val_met.get("ibs", float("nan"))),
            "train_auc_mean": float(train_met.get("auc_mean", float("nan"))),
            "val_auc_mean": float(val_met.get("auc_mean", float("nan"))),
            "train_nb_mean": float(train_met.get("nb_mean", float("nan"))),
            "val_nb_mean": float(val_met.get("nb_mean", float("nan"))),
        }
        for tday in args.auc_times_days:
            key = f"auc_{int(round(float(tday)))}d"
            row[f"train_{key}"] = float(train_met.get(key, float("nan")))
            row[f"val_{key}"] = float(val_met.get(key, float("nan")))
        pd.DataFrame([row]).to_csv(metrics_csv, mode="a", header=not metrics_csv.is_file(), index=False)

        save_checkpoint(
            ckpt_last,
            epoch=int(epoch),
            num_time_bins=int(fold_num_time_bins),
            model=model,
            args=args,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            report_metric=report_metric,
            ema=ema,
            swa=swa,
        )
        if np.isfinite(report_metric) and report_metric > best_report_metric:
            save_checkpoint(
                ckpt_best,
                epoch=int(epoch),
                num_time_bins=int(fold_num_time_bins),
                model=model,
                args=args,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                report_metric=report_metric,
                ema=ema,
                swa=swa,
            )
            best_report_metric = float(report_metric)
            print(f"[fold {fold:02d}] new best {args.report_metric}={best_report_metric:.4f} -> {ckpt_best}")
        scheduler.step()

    last_test = evaluate_model(
        model, te_loader, device,
        endpoint=str(args.endpoint),
        risk_horizon_days=float(args.risk_horizon_days),
        time_bin_width_days=float(args.time_bin_width_days),
        eval_times_days=args.auc_times_days,
        dca_thresholds=args.dca_thresholds,
        autocast_ctx=autocast_ctx,
    )
    risks_last = predict_risk_scores(model, te_loader, device, endpoint=str(args.endpoint), risk_horizon_days=float(args.risk_horizon_days), autocast_ctx=autocast_ctx)

    ema_test = None
    risks_ema = None
    if ema is not None:
        with ema.apply_to(mm):
            ema_test = evaluate_model(
                model, te_loader, device,
                endpoint=str(args.endpoint),
                risk_horizon_days=float(args.risk_horizon_days),
                time_bin_width_days=float(args.time_bin_width_days),
                eval_times_days=args.auc_times_days,
                dca_thresholds=args.dca_thresholds,
                autocast_ctx=autocast_ctx,
            )
            risks_ema = predict_risk_scores(model, te_loader, device, endpoint=str(args.endpoint), risk_horizon_days=float(args.risk_horizon_days), autocast_ctx=autocast_ctx)

    swa_test = None
    risks_swa = None
    if swa is not None and swa.n_averaged > 0:
        with swa.apply_to(mm):
            swa_test = evaluate_model(
                model, te_loader, device,
                endpoint=str(args.endpoint),
                risk_horizon_days=float(args.risk_horizon_days),
                time_bin_width_days=float(args.time_bin_width_days),
                eval_times_days=args.auc_times_days,
                dca_thresholds=args.dca_thresholds,
                autocast_ctx=autocast_ctx,
            )
            risks_swa = predict_risk_scores(model, te_loader, device, endpoint=str(args.endpoint), risk_horizon_days=float(args.risk_horizon_days), autocast_ctx=autocast_ctx)

    best_test = None
    risks_best = None
    if load_model_state_only(ckpt_best, model):
        best_test = evaluate_model(
            model, te_loader, device,
            endpoint=str(args.endpoint),
            risk_horizon_days=float(args.risk_horizon_days),
            time_bin_width_days=float(args.time_bin_width_days),
            eval_times_days=args.auc_times_days,
            dca_thresholds=args.dca_thresholds,
            autocast_ctx=autocast_ctx,
        )
        risks_best = predict_risk_scores(model, te_loader, device, endpoint=str(args.endpoint), risk_horizon_days=float(args.risk_horizon_days), autocast_ctx=autocast_ctx)

    export_suffix = "ema" if (risks_ema is not None) else "last"
    export_risks = risks_ema if (risks_ema is not None) else risks_last
    save_endpoint_risk_dict_csv(
        export_risks,
        fold_dir / f"test_risks_{export_suffix}.csv",
        id_col=args.id_col,
        endpoint=str(args.endpoint),
        risk_horizon_days=float(args.risk_horizon_days),
    )

    if risks_best is not None:
        save_endpoint_risk_dict_csv(
            risks_best,
            fold_dir / "test_risks_best.csv",
            id_col=args.id_col,
            endpoint=str(args.endpoint),
            risk_horizon_days=float(args.risk_horizon_days),
        )

    if args.export_extra_risks:
        save_endpoint_risk_dict_csv(
            risks_last,
            fold_dir / "test_risks_last.csv",
            id_col=args.id_col,
            endpoint=str(args.endpoint),
            risk_horizon_days=float(args.risk_horizon_days),
        )
        if risks_ema is not None:
            save_endpoint_risk_dict_csv(
                risks_ema,
                fold_dir / "test_risks_ema.csv",
                id_col=args.id_col,
                endpoint=str(args.endpoint),
                risk_horizon_days=float(args.risk_horizon_days),
            )
        if risks_swa is not None:
            save_endpoint_risk_dict_csv(
                risks_swa,
                fold_dir / "test_risks_swa.csv",
                id_col=args.id_col,
                endpoint=str(args.endpoint),
                risk_horizon_days=float(args.risk_horizon_days),
            )

    met_export = ema_test if (export_suffix == "ema") else last_test
    return {
        "fold": int(fold),
        "export_suffix": export_suffix,
        "export_risks": export_risks,
        "test_metrics_export": met_export,
        "test_metrics_last": last_test,
        "test_metrics_ema": ema_test,
        "test_metrics_swa": swa_test,
        "test_metrics_best": best_test,
        "n_test": int(len(te_df)),
        "num_time_bins": int(fold_num_time_bins),
        "train_max_time_days": float(fold_max_time),
    }


# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--meta_csv", required=True)
    p.add_argument("--id_col", type=str, default="patient_id")
    p.add_argument("--ct_col", type=str, default="ct_out_path")

    p.add_argument("--mask_pt_col", type=str, default="mask_pt_out_path")
    p.add_argument("--mask_ln_col", type=str, default="mask_ln_out_path")

    p.add_argument("--endpoint", type=str, default="OS", choices=["OS", "DSS", "DFS"])
    p.add_argument("--time_col", type=str, default="")
    p.add_argument("--event_col", type=str, default="")

    p.add_argument("--keep_bad_status", action="store_true")
    p.add_argument("--keep_unmatched_survival", action="store_true")

    p.add_argument("--splits_dir", type=str, default="")
    p.add_argument("--splits_csv", type=str, default="")
    p.add_argument("--cv_folds", type=int, default=4)
    p.add_argument("--debug_fold", type=int, default=-1)
    p.add_argument("--strict_splits", action="store_true")

    p.add_argument("--out_dir", type=str, default="runs/contour_aware_survival")
    p.add_argument("--exp_name", type=str, default="contour_aware_survival")

    p.add_argument("--device", type=str, default="")
    p.add_argument("--data_parallel", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log_every_batches", type=int, default=50)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--resume", dest="resume", action="store_true")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.set_defaults(resume=True)

    p.add_argument("--min_lr", type=float, default=1e-6)

    p.add_argument("--debug_max_train", type=int, default=-1)
    p.add_argument("--debug_max_val", type=int, default=-1)
    p.add_argument("--debug_max_test", type=int, default=-1)

    p.add_argument("--strict_files", dest="strict_files", action="store_true")
    p.add_argument("--no_strict_files", dest="strict_files", action="store_false")
    p.set_defaults(strict_files=True)

    # LRs/WDs
    p.add_argument("--lr_backbone", type=float, default=1e-3)
    p.add_argument("--lr_lora", type=float, default=5e-4)
    p.add_argument("--lr_head", type=float, default=1e-4)

    p.add_argument("--wd_backbone", type=float, default=0.0)
    p.add_argument("--wd_lora", type=float, default=0.0)
    p.add_argument("--wd_head", type=float, default=1e-3)
    p.add_argument("--wd_clin", type=float, default=5e-4)
    p.add_argument("--wd_rad", type=float, default=1e-3)

    # Dropouts/lambdas
    p.add_argument("--expert_dropout_p", type=float, default=0.15)
    p.add_argument("--gate_entropy_lambda", type=float, default=1e-2)
    p.add_argument("--gate_loadbal_lambda", type=float, default=1e-2)
    p.add_argument("--proj_dropout_p", type=float, default=0.30)
    p.add_argument("--attn_dropout_p", type=float, default=0.15)
    p.add_argument("--gate_dropout_p", type=float, default=0.20)
    p.add_argument("--surv_dropout_p", type=float, default=0.40)
    p.add_argument("--hazard_smooth_lambda", type=float, default=1e-2)
    p.add_argument("--logit_l2_lambda", type=float, default=1e-5)

    p.add_argument("--clinical_noise_std", type=float, default=0.02)
    p.add_argument("--radiomics_noise_std", type=float, default=0.02)
    p.add_argument("--modality_dropout_clin_p", type=float, default=0.20)
    p.add_argument("--modality_dropout_rad_p", type=float, default=0.25)

    p.add_argument("--time_bin_width_days", type=float, default=180.0)
    p.add_argument("--max_time_bins", type=int, default=100)
    p.add_argument("--risk_horizon_days", type=float, default=3 * 365.0)
    p.add_argument("--auc_times_days", type=float, nargs="*", default=[365.0, 3 * 365.0, 5 * 365.0])
    p.add_argument("--dca_thresholds", type=float, nargs="*", default=[0.1, 0.2, 0.3])
    p.add_argument("--report_metric", type=str, default="auc_1095d")
    p.add_argument("--primary_surv_loss_weight", type=float, default=1.0)
    p.add_argument("--aux_surv_loss_weight", type=float, default=0.35)

    p.add_argument("--use_ema", dest="use_ema", action="store_true")
    p.add_argument("--no_ema", dest="use_ema", action="store_false")
    p.set_defaults(use_ema=True)
    p.add_argument("--ema_decay", type=float, default=0.995)

    p.add_argument("--use_swa", dest="use_swa", action="store_true")
    p.add_argument("--no_swa", dest="use_swa", action="store_false")
    p.set_defaults(use_swa=True)
    p.add_argument("--swa_start_epoch", type=int, default=-1)
    p.add_argument("--swa_update_freq_epochs", type=int, default=1)

    p.add_argument("--export_extra_risks", dest="export_extra_risks", action="store_true")
    p.add_argument("--no_export_extra_risks", dest="export_extra_risks", action="store_false")
    p.set_defaults(export_extra_risks=True)

    p.add_argument("--clinical_cols", type=str, nargs="*", default=DEFAULT_CLINICAL_COLS)

    p.add_argument("--use_radiomics", dest="use_radiomics", action="store_true")
    p.add_argument("--no_radiomics", dest="use_radiomics", action="store_false")
    p.set_defaults(use_radiomics=True)
    p.add_argument(
        "--radiomics_root",
        type=str,
        default="radiomics_features/radiomics_features",
        help="Radiomics source: directory of per-patient CSVs or a patient-wide CSV file.",
    )
    p.add_argument("--radiomics_pca_total_components", type=int, default=100)

    # Swin cfg
    p.add_argument("--img_size", type=int, nargs=3, default=[128, 256, 256])
    p.add_argument("--feature_size", type=int, default=96)
    p.add_argument("--depths", type=int, nargs=4, default=[2, 2, 18, 2])
    p.add_argument("--num_heads", type=int, nargs=4, default=[3, 6, 12, 24])
    p.add_argument("--drop_rate", type=float, default=0.0)
    p.add_argument("--attn_drop_rate", type=float, default=0.0)
    p.add_argument("--dropout_path_rate", type=float, default=0.0)
    p.add_argument("--use_checkpoint", dest="use_checkpoint", action="store_true")
    p.add_argument("--no_use_checkpoint", dest="use_checkpoint", action="store_false")
    p.set_defaults(use_checkpoint=False)
    p.add_argument("--image_encoder_mode", type=str, default="contour_aware", choices=["contour_aware"], help="CT-only shared encoder with internal PT/LN localization heads and soft-mask ROI tokenization.")

    p.add_argument("--align_swin_cfg_from_contour_warmstart", dest="align_swin_cfg_from_contour_warmstart", action="store_true")
    p.add_argument("--no_align_swin_cfg_from_contour_warmstart", dest="align_swin_cfg_from_contour_warmstart", action="store_false")
    p.add_argument("--align_swin_cfg_from_seg_ckpt", dest="align_swin_cfg_from_contour_warmstart", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--no_align_swin_cfg_from_seg_ckpt", dest="align_swin_cfg_from_contour_warmstart", action="store_false", help=argparse.SUPPRESS)
    p.set_defaults(align_swin_cfg_from_contour_warmstart=True)

    # fusion dims
    p.add_argument("--fused_dim", type=int, default=512)

    # token backbone knobs (image)
    p.add_argument("--img_token_dim", type=int, default=0)
    p.add_argument("--token_mlp_dropout", type=float, default=0.40)
    p.add_argument("--token_mlp_hidden_dim", type=int, default=0)
    p.add_argument("--token_dropout", type=float, default=0.10)

    p.add_argument("--attn_mask_bias", type=float, default=2.0)
    p.add_argument("--use_multiscale", action="store_true")
    p.add_argument("--mask_interp", type=str, default="nearest", choices=["nearest", "trilinear"])
    p.add_argument("--min_roi_frac", type=float, default=1e-5)
    p.add_argument("--min_roi_voxels_deep", type=int, default=8)

    # image/rad head capacity knobs
    p.add_argument("--img_proj_hidden_dim", type=int, default=0)
    p.add_argument("--img_tok_ffn_hidden_dim", type=int, default=0)
    p.add_argument("--img_post_hidden_dim", type=int, default=0)
    p.add_argument("--img_attn_heads", type=int, default=4)
    p.add_argument("--gate_hidden_dim", type=int, default=0)
    p.add_argument("--rad_hidden_dim", type=int, default=0)
    p.add_argument("--rad_proj_dropout_p", type=float, default=0.50)

    p.add_argument("--pt_shell_radius", type=int, default=3)
    p.add_argument("--ln_shell_radius", type=int, default=3)
    p.add_argument("--teacher_force_epochs", type=int, default=12)
    p.add_argument("--teacher_force_start", type=float, default=1.0)
    p.add_argument("--teacher_force_end", type=float, default=0.0)
    p.add_argument("--loc_loss_pt_lambda", type=float, default=0.25)
    p.add_argument("--loc_loss_ln_lambda", type=float, default=0.25)
    p.add_argument("--loc_presence_lambda", type=float, default=0.05)
    p.add_argument("--loc_bce_weight", type=float, default=0.5)
    p.add_argument("--loc_dice_weight", type=float, default=0.5)

    p.add_argument("--shell_body_from_ct", action="store_true")
    p.add_argument("--body_ct_thr", type=str, default="auto")
    p.add_argument("--body_ct_thr_hu", type=float, default=-500.0)
    p.add_argument("--body_close_r", type=int, default=2)
    p.add_argument("--body_max_frac", type=float, default=0.995)

    p.add_argument("--strict_swinvit_layout", dest="strict_swinvit_layout", action="store_true")
    p.add_argument("--no_strict_swinvit_layout", dest="strict_swinvit_layout", action="store_false")
    p.set_defaults(strict_swinvit_layout=True)
    p.add_argument("--debug_swinvit_layout", action="store_true")

    p.add_argument("--contour_warmstart_ckpt", type=str, default="")
    p.add_argument("--contour_warmstart_dir", type=str, default="")
    p.add_argument("--contour_warmstart_name", type=str, default="best.pt")
    p.add_argument("--shared_seg_pretrain_ckpt", dest="contour_warmstart_ckpt", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--shared_seg_pretrain_dir", dest="contour_warmstart_dir", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--shared_seg_pretrain_name", dest="contour_warmstart_name", type=str, default="best.pt", help=argparse.SUPPRESS)

    # LoRA
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_scope", type=str, default="shared", choices=["shared"])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=32.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, nargs="*", default=["qkv", "proj", "fc1", "fc2"])
    p.add_argument("--lora_min_replacements", type=int, default=1)

    # Debug / safety
    p.add_argument("--nan_guard", action="store_true")
    p.add_argument("--print_trainable_backbone_params", action="store_true")

    args = p.parse_args()

    if args.time_col == "" or args.event_col == "":
        tcol, ecol = ENDPOINT_MAP[args.endpoint]
        if args.time_col == "":
            args.time_col = tcol
        if args.event_col == "":
            args.event_col = ecol

    if bool(args.splits_dir) and bool(args.splits_csv):
        raise ValueError("Provide only one of --splits_dir or --splits_csv.")
    if int(args.swa_start_epoch) < 0:
        args.swa_start_epoch = int(0.75 * int(args.epochs))

    if int(args.img_token_dim) <= 0:
        args.img_token_dim = int(args.fused_dim)
    if int(args.token_mlp_hidden_dim) <= 0:
        args.token_mlp_hidden_dim = int(2 * int(args.img_token_dim))
    if int(args.img_proj_hidden_dim) <= 0:
        args.img_proj_hidden_dim = int(2 * int(args.fused_dim))
    if int(args.img_tok_ffn_hidden_dim) <= 0:
        args.img_tok_ffn_hidden_dim = int(2 * int(args.fused_dim))
    if int(args.img_post_hidden_dim) <= 0:
        args.img_post_hidden_dim = int(2 * int(args.fused_dim))
    if int(args.gate_hidden_dim) <= 0:
        args.gate_hidden_dim = int(args.fused_dim)
    if int(args.rad_hidden_dim) <= 0:
        args.rad_hidden_dim = int(max(512, 2 * int(args.fused_dim)))
    if float(args.primary_surv_loss_weight) < 0.0 or float(args.aux_surv_loss_weight) < 0.0:
        raise ValueError("--primary_surv_loss_weight and --aux_surv_loss_weight must be >= 0.")

    args.auc_times_days = [float(x) for x in (args.auc_times_days or [])]
    args.dca_thresholds = [float(x) for x in (args.dca_thresholds or [])]

    if bool(args.align_swin_cfg_from_contour_warmstart):
        args = align_backbone_cfg_to_contour_warmstart(args)

    return args


def _validate_event_column(meta: pd.DataFrame, event_col: str):
    vals = sorted(set(pd.to_numeric(meta[event_col], errors="coerce").dropna().astype(int).tolist()))
    bad = [v for v in vals if v not in (0, 1)]
    if bad:
        raise RuntimeError(f"[DATA] {event_col} must be binary {{0,1}}. Found other values: {bad[:20]}")


def compute_multitask_num_time_bins(meta: pd.DataFrame, args) -> Tuple[int, float]:
    max_bins = 1
    max_time = 0.0
    for time_col in MULTITASK_TIME_COLS:
        if time_col not in meta.columns:
            continue
        bins, col_max_time = H.compute_global_num_time_bins(
            meta,
            time_col=time_col,
            time_bin_width_days=float(args.time_bin_width_days),
            max_time_bins=int(args.max_time_bins),
            risk_horizon_days=float(args.risk_horizon_days),
            auc_times_days=args.auc_times_days,
        )
        max_bins = max(max_bins, int(bins))
        max_time = max(max_time, float(col_max_time))
    return int(max_bins), float(max_time)

def main():
    _configure_stdio_line_buffering()
    args = parse_args()
    set_seed(args.seed)

    device = parse_device(args.device)
    bind_cuda_device(device)

    scaler, autocast_ctx = make_amp(device, enabled=bool(args.amp))
    print(f"[info] device={device} amp={bool(args.amp and device.type=='cuda')} use_lora={bool(args.use_lora)}")
    if not bool(args.strict_files):
        print("[warn] strict_files is disabled; missing CT/PT/LN files will be replaced with zero arrays.")
    print(
        "[mtl] survival_heads="
        f"{'/'.join(SURVIVAL_ENDPOINTS)} "
        f"primary={str(args.endpoint).upper()} "
        f"primary_w={float(args.primary_surv_loss_weight):.2f} "
        f"aux_w={float(args.aux_surv_loss_weight):.2f}"
    )

    out_root = Path(args.out_dir) / args.exp_name
    out_root.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta_csv, dtype={args.id_col: str})
    meta[args.id_col] = meta[args.id_col].astype(str)
    data_root = str(Path(args.meta_csv).resolve().parent)

    if (not args.keep_bad_status) and ("status" in meta.columns):
        meta = meta[meta["status"].astype(str).str.lower() == "ok"].copy()
    for time_col in MULTITASK_TIME_COLS:
        if time_col in meta.columns:
            meta[time_col] = pd.to_numeric(meta[time_col], errors="coerce")
    for event_col in MULTITASK_EVENT_COLS:
        if event_col in meta.columns:
            meta[event_col] = pd.to_numeric(meta[event_col], errors="coerce")

    for event_col in MULTITASK_EVENT_COLS:
        if event_col not in meta.columns:
            continue
        valid_ep = meta[event_col].notna()
        if bool(valid_ep.any()):
            _validate_event_column(meta.loc[valid_ep].copy(), event_col)
            meta.loc[valid_ep, event_col] = meta.loc[valid_ep, event_col].astype(int)

    valid_any = pd.Series(False, index=meta.index)
    for time_col, event_col in zip(MULTITASK_TIME_COLS, MULTITASK_EVENT_COLS):
        if time_col not in meta.columns or event_col not in meta.columns:
            continue
        valid_any = valid_any | _valid_survival_mask_frame(meta, time_col, event_col)
    meta = meta.loc[valid_any].copy()
    primary_valid = _valid_survival_mask_frame(meta, args.time_col, args.event_col)
    if not bool(primary_valid.any()):
        raise RuntimeError(f"[DATA] no rows with valid primary endpoint labels for endpoint={args.endpoint}")

    resolved_img_size, data_shape = resolve_img_size_against_data(
        meta,
        args.ct_col,
        args.img_size,
        id_col=args.id_col,
        data_root=data_root,
    )
    args.img_size = list(resolved_img_size)
    print(f"[IMGCFG] data_shape(D,H,W)={data_shape} | using img_size(D,H,W)={tuple(args.img_size)}")

    splits = load_precomputed_splits(args.cv_folds, splits_dir=args.splits_dir, splits_csv=args.splits_csv)
    folds = [int(args.debug_fold)] if int(args.debug_fold) >= 0 else list(range(int(args.cv_folds)))

    fold_results: List[Dict[str, Any]] = []
    all_test_risks: Dict[str, float] = {}

    for f in folds:
        res = run_one_fold(
            fold=int(f),
            args=args,
            meta=meta,
            split=splits[int(f)],
            device=device,
            scaler=scaler,
            autocast_ctx=autocast_ctx,
            out_root=out_root,
        )
        dup = set(all_test_risks).intersection(res["export_risks"])
        if dup:
            raise RuntimeError(f"[CV] duplicate test IDs across folds: {sorted(list(dup))[:20]}")
        fold_results.append(res)
        all_test_risks.update(res["export_risks"])

    if all_test_risks:
        save_endpoint_risk_dict_csv(
            all_test_risks,
            out_root / "cv_test_risks_export.csv",
            id_col=args.id_col,
            endpoint=str(args.endpoint),
            risk_horizon_days=float(args.risk_horizon_days),
        )

    test_c = []
    for r in fold_results:
        met = r["test_metrics_export"] or {}
        test_c.append(float(met.get("c_index", float("nan"))))

    summary = {
        "exp_name": args.exp_name,
        "folds_run": folds,
        "fold_num_time_bins": [int(r.get("num_time_bins", 0)) for r in fold_results],
        "fold_train_max_time_days": [float(r.get("train_max_time_days", float("nan"))) for r in fold_results],
        "mean_fold_test_c_index": float(np.nanmean(np.array(test_c, dtype=float))) if test_c else float("nan"),
        "risk_horizon_days": float(args.risk_horizon_days),
        "primary_endpoint": str(args.endpoint),
        "survival_heads": list(SURVIVAL_ENDPOINTS),
        "primary_surv_loss_weight": float(args.primary_surv_loss_weight),
        "aux_surv_loss_weight": float(args.aux_surv_loss_weight),
        "device": str(device),
        "use_lora": bool(args.use_lora),
        "lora_scope": str(args.lora_scope),
        "lora_r": int(args.lora_r),
        "lora_alpha": float(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_targets": list(args.lora_targets or []),
        "teacher_force_epochs": int(args.teacher_force_epochs),
        "teacher_force_start": float(args.teacher_force_start),
        "teacher_force_end": float(args.teacher_force_end),
        "loc_loss_pt_lambda": float(args.loc_loss_pt_lambda),
        "loc_loss_ln_lambda": float(args.loc_loss_ln_lambda),
        "loc_presence_lambda": float(args.loc_presence_lambda),
        "nan_guard": bool(args.nan_guard),
    }
    with open(out_root / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== CV Summary ===")
    print(f"folds_run: {folds}")
    print(f"mean fold test c-index: {summary['mean_fold_test_c_index']:.4f}")
    print(f"[done] wrote {out_root/'cv_summary.json'}")


if __name__ == "__main__":
    main()
