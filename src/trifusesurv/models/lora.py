"""LoRA (Low-Rank Adaptation) utilities for TriFuseSurv."""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# LoRA linear wrapper
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable low-rank adapters.

    y = Wx + b + (alpha/r) * B(A(drop(x)))
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)

        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = float(lora_alpha / max(1, r)) if r > 0 else 0.0
        self.drop = (
            nn.Dropout(p=float(lora_dropout))
            if float(lora_dropout) > 0
            else nn.Identity()
        )

        # Freeze base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        dev = self.base.weight.device
        dt = self.base.weight.dtype

        if self.r > 0:
            self.lora_A = nn.Linear(self.in_features, self.r, bias=False).to(device=dev, dtype=dt)
            self.lora_B = nn.Linear(self.r, self.out_features, bias=False).to(device=dev, dtype=dt)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r <= 0:
            return y
        z = self.lora_B(self.lora_A(self.drop(x)))
        return y + (self.scaling * z)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_parent_module(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _normalize_lora_targets(target_keywords: Sequence[str]):
    out = set()
    for k in target_keywords:
        k = str(k).strip().lower()
        if k:
            out.add(k)
    return sorted(out)


def is_lora_param_name(n: str) -> bool:
    ln = n.lower()
    return ".lora_a." in ln or ".lora_b." in ln or ln.endswith(".lora_a.weight") or ln.endswith(".lora_b.weight")


# ---------------------------------------------------------------------------
# Injection: keyword-based (for training)
# ---------------------------------------------------------------------------
def inject_lora_into_module(
    root: nn.Module,
    *,
    target_keywords: Sequence[str],
    r: int,
    alpha: float,
    dropout: float,
    verbose: bool = True,
) -> int:
    """Inject LoRA wrappers into matching nn.Linear modules (keyword-based)."""
    keys = _normalize_lora_targets(target_keywords)
    named = list(root.named_modules())
    nrep = 0
    for name, mod in named:
        if not name:
            continue
        if not isinstance(mod, nn.Linear):
            continue
        lname = name.lower()
        if not any(k in lname for k in keys):
            continue

        parent, child = _get_parent_module(root, name)
        cur = getattr(parent, child)
        if isinstance(cur, LoRALinear):
            continue

        lora_mod = LoRALinear(cur, r=int(r), lora_alpha=float(alpha), lora_dropout=float(dropout))
        lora_mod.to(device=cur.weight.device, dtype=cur.weight.dtype)
        setattr(parent, child, lora_mod)
        nrep += 1

    if verbose:
        print(f"[LoRA] injected {nrep} LoRA Linear(s) into {root.__class__.__name__} targets={keys} r={r} alpha={alpha} drop={dropout}")
    return nrep


# ---------------------------------------------------------------------------
# Injection: checkpoint-based (for inference / SHAP)
# ---------------------------------------------------------------------------
def inject_lora_from_state_dict(
    root: nn.Module,
    sd: Dict[str, torch.Tensor],
    *,
    lora_alpha: float,
    lora_dropout: float,
    scope: str = "both",
    verbose: bool = True,
) -> int:
    """Inject LoRA wrappers where checkpoint expects them (by scanning keys)."""
    scope = str(scope).lower().strip()
    if scope not in ("pt", "ln", "both", "shared", "all", "auto"):
        raise ValueError(f"--lora_scope must be pt|ln|both|shared|all|auto, got {scope}")

    prefixes = sorted({
        k[: -len(".lora_A.weight")]
        for k in sd.keys()
        if k.endswith(".lora_A.weight")
    })
    if not prefixes:
        return 0

    def _scope_matches(prefix: str) -> bool:
        if scope in ("all", "auto"):
            return True
        if scope == "pt":
            return ".backbone_pt." in prefix
        if scope == "ln":
            return ".backbone_ln." in prefix
        if scope == "shared":
            return ".backbone_shared." in prefix
        # "both" historically meant both PT/LN branches; in the shared-mask
        # architecture there is only one image backbone, so accept any image branch.
        return any(tag in prefix for tag in (".backbone_pt.", ".backbone_ln.", ".backbone_shared."))

    nrep = 0
    for pref in prefixes:
        if not _scope_matches(pref):
            continue

        akey = pref + ".lora_A.weight"
        if akey not in sd:
            continue
        r = int(sd[akey].shape[0])

        parent, child = _get_parent_module(root, pref)
        cur = getattr(parent, child)

        if isinstance(cur, LoRALinear):
            continue
        if not isinstance(cur, nn.Linear):
            raise RuntimeError(
                f"[LoRA] Checkpoint expects LoRA at '{pref}', but model has {type(cur)}. "
                f"Architecture mismatch."
            )

        lora_mod = LoRALinear(cur, r=r, lora_alpha=float(lora_alpha), lora_dropout=float(lora_dropout))
        lora_mod.to(device=cur.weight.device, dtype=cur.weight.dtype)
        setattr(parent, child, lora_mod)
        nrep += 1

    if verbose:
        print(f"[LoRA] injected {nrep} LoRA module(s) | scope={scope} alpha={lora_alpha} drop={lora_dropout}")
    return nrep


# ---------------------------------------------------------------------------
# Freeze / unfreeze utilities
# ---------------------------------------------------------------------------
def freeze_all_params(mod: nn.Module):
    for p in mod.parameters():
        p.requires_grad = False


def mark_only_lora_trainable(mod: nn.Module):
    for n, p in mod.named_parameters():
        p.requires_grad = is_lora_param_name(n)


def count_trainable(mod: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in mod.parameters())
    trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    return total, trainable
