import contextlib
import math
from typing import Tuple, Dict, List, Optional, Any, Sequence

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "get_module_param_dtype",
    "set_seed",
    "seed_worker",
    "compute_global_num_time_bins",
    "EMAWeights",
    "SWAWeights",
    "discrete_time_nll_loss",
    "hazard_smoothness_penalty",
    "hazards_to_survival_end_of_bin_numpy",
    "survival_at_time_from_hazards",
    "risk_at_time_from_hazards",
    "concordance_index",
    "integrated_brier_score",
    "time_dependent_auc_surv",
    "decision_curve_analysis_surv",
    "ipcw_auc_at_horizon_from_risk",
]


# =============================================================================
# Small utilities
# =============================================================================
def get_module_param_dtype(module: nn.Module, default: torch.dtype = torch.float32) -> torch.dtype:
    for p in module.parameters():
        return p.dtype
    return default


def set_seed(seed: int = 42):
    seed = int(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    import random
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# Time bins
# =============================================================================
def compute_global_num_time_bins(
    df: pd.DataFrame,
    time_col: str,
    time_bin_width_days: float,
    max_time_bins: int,
    risk_horizon_days: float,
    auc_times_days: Sequence[float],
) -> Tuple[int, float]:
    """
    Compute a SINGLE global discrete-time bin count shared across all folds/runs.

    IMPORTANT: this code assumes `time_col` is in DAYS.

    Definitions:
      - Bin k corresponds to time interval [k*bw, (k+1)*bw).
      - NLL uses bin_idx = floor(t / bw), so to cover times up to max_time we need:
          K >= floor(max_time/bw) + 1
      - Exact-time horizon metrics require the horizon falls within [0, K*bw]:
          K >= ceil(horizon/bw)
    """
    times = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
    times = times[np.isfinite(times)]
    global_max_time = float(times.max()) if times.size else 0.0

    bw = float(time_bin_width_days)
    bw = max(bw, 1e-6)

    bins_by_data = int(np.floor(global_max_time / bw)) + 1 if global_max_time > 0 else 1

    max_auc_time = float(max(auc_times_days)) if auc_times_days else 0.0
    bins_by_horizon = int(np.ceil(float(risk_horizon_days) / bw)) if risk_horizon_days > 0 else 1
    bins_by_auc = int(np.ceil(max_auc_time / bw)) if max_auc_time > 0 else 1

    num_time_bins = max(1, bins_by_data, bins_by_horizon, bins_by_auc)
    num_time_bins = int(min(num_time_bins, int(max_time_bins)))

    needed = max(bins_by_horizon, bins_by_auc)
    if num_time_bins < needed:
        print(
            f"[TIME][WARN] Global num_time_bins={num_time_bins} capped by MAX_TIME_BINS={max_time_bins} "
            f"and may not fully cover requested horizons (needed={needed})."
        )

    return num_time_bins, global_max_time


# =============================================================================
# EMA / SWA
# =============================================================================
class EMAWeights:
    """
    Exponential moving average of parameters.
    Tracks either all params or only trainable params (recommended).
    """
    def __init__(self, model: nn.Module, decay: float = 0.999, track_trainable_only: bool = True):
        self.decay = float(decay)
        self.track_trainable_only = bool(track_trainable_only)
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self._init_from_model(model)

    def _init_from_model(self, model: nn.Module):
        self.shadow.clear()
        for name, p in model.named_parameters():
            if self.track_trainable_only and (not p.requires_grad):
                continue
            self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1
        d = float(self.decay)
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "decay": float(self.decay),
            "track_trainable_only": bool(self.track_trainable_only),
            "num_updates": int(self.num_updates),
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any], model: nn.Module):
        self.decay = float(state.get("decay", self.decay))
        self.track_trainable_only = bool(state.get("track_trainable_only", self.track_trainable_only))
        self.num_updates = int(state.get("num_updates", 0))

        shadow_in = state.get("shadow", {})
        self.shadow.clear()

        param_map = {n: p for n, p in model.named_parameters()}
        for name, tens_cpu in shadow_in.items():
            if name not in param_map:
                continue
            p = param_map[name]
            self.shadow[name] = tens_cpu.to(device=p.device, dtype=p.dtype)

    @contextlib.contextmanager
    def apply_to(self, model: nn.Module):
        backups: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for name, p in model.named_parameters():
                if name not in self.shadow:
                    continue
                backups.append((p, p.detach().clone()))
                p.data.copy_(self.shadow[name])
            yield
        finally:
            for p, old in backups:
                p.data.copy_(old)


class SWAWeights:
    """
    Simple SWA (uniform average) over epochs.
    Tracks either all params or only trainable params (recommended).
    """
    def __init__(self, model: nn.Module, track_trainable_only: bool = True):
        self.track_trainable_only = bool(track_trainable_only)
        self.n_averaged = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        self._param_names: List[str] = []
        self._capture_param_names(model)

    def _capture_param_names(self, model: nn.Module):
        self._param_names = []
        for name, p in model.named_parameters():
            if self.track_trainable_only and (not p.requires_grad):
                continue
            self._param_names.append(name)

    @torch.no_grad()
    def update(self, model: nn.Module):
        if not self.shadow:
            for n, p in model.named_parameters():
                if n in self._param_names:
                    self.shadow[n] = p.detach().clone()
            self.n_averaged = 1
            return

        self.n_averaged += 1
        n = float(self.n_averaged)
        alpha = 1.0 / n
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(1.0 - alpha).add_(p.detach(), alpha=alpha)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "track_trainable_only": bool(self.track_trainable_only),
            "n_averaged": int(self.n_averaged),
            "param_names": list(self._param_names),
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state: Dict[str, Any], model: nn.Module):
        self.track_trainable_only = bool(state.get("track_trainable_only", self.track_trainable_only))
        self.n_averaged = int(state.get("n_averaged", 0))
        self._param_names = list(state.get("param_names", self._param_names))

        shadow_in = state.get("shadow", {})
        self.shadow.clear()

        param_map = {n: p for n, p in model.named_parameters()}
        for name, tens_cpu in shadow_in.items():
            if name not in param_map:
                continue
            p = param_map[name]
            self.shadow[name] = tens_cpu.to(device=p.device, dtype=p.dtype)

    @contextlib.contextmanager
    def apply_to(self, model: nn.Module):
        backups: List[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for name, p in model.named_parameters():
                if name not in self.shadow:
                    continue
                backups.append((p, p.detach().clone()))
                p.data.copy_(self.shadow[name])
            yield
        finally:
            for p, old in backups:
                p.data.copy_(old)


# =============================================================================
# Loss
# =============================================================================
def discrete_time_nll_loss(
    hazards_logits: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    time_bin_width: float,
    num_time_bins: int,
) -> torch.Tensor:
    """
    Discrete-time survival negative log-likelihood.

    Conventions:
      - hazards_logits[b,k] is the logit of conditional hazard in bin k.
      - bin_idx = floor(t/bw), clamped into [0, K-1]
      - If event:   loglik = sum_{j<k} log(1-h_j) + log(h_k)
      - If censored: loglik = sum_{j<=k} log(1-h_j)  (censor-at-end-of-bin convention)
    """
    if hazards_logits.ndim != 2:
        raise ValueError(f"hazards_logits must be (B,K), got {hazards_logits.shape}")
    B, K = hazards_logits.shape
    if int(K) != int(num_time_bins):
        raise ValueError(f"num_time_bins={num_time_bins} but hazards_logits.shape[1]={K}")
    if times.shape[0] != B or events.shape[0] != B:
        raise ValueError("times/events must have same batch size as hazards_logits")

    bw = float(time_bin_width)
    bw = max(bw, 1e-6)

    device = hazards_logits.device
    times = times.to(device).view(-1)
    events = events.to(device).view(-1).float()

    bin_idx = torch.floor(times / bw).long().clamp(0, int(num_time_bins) - 1)

    log_h = -F.softplus(-hazards_logits)          # log(sigmoid)
    log1m_h = F.logsigmoid(-hazards_logits)       # log(1 - sigmoid)

    cum_log1m = torch.cumsum(log1m_h, dim=1)      # (B,K)

    is_event = (events == 1.0)
    is_cens = ~is_event

    total_loglik = hazards_logits.new_zeros(())

    if is_event.any():
        event_rows = torch.nonzero(is_event, as_tuple=False).view(-1)
        k = bin_idx[is_event]
        pre = torch.zeros_like(k, dtype=hazards_logits.dtype, device=device)
        has_pre = (k > 0)
        if has_pre.any():
            pre_rows = event_rows[has_pre]
            pre_cols = (k - 1)[has_pre]
            pre[has_pre] = cum_log1m[pre_rows, pre_cols]
        ll = pre + log_h[event_rows, k]
        total_loglik = total_loglik + ll.sum()

    if is_cens.any():
        cens_rows = torch.nonzero(is_cens, as_tuple=False).view(-1)
        k = bin_idx[is_cens]
        ll = cum_log1m[cens_rows, k]
        total_loglik = total_loglik + ll.sum()

    loss = -total_loglik / float(B)
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def hazard_smoothness_penalty(hazards_logits: torch.Tensor) -> torch.Tensor:
    if hazards_logits.ndim != 2 or hazards_logits.size(1) < 2:
        return hazards_logits.new_tensor(0.0)
    h = torch.sigmoid(hazards_logits)
    return ((h[:, 1:] - h[:, :-1]) ** 2).mean()


# =============================================================================
# Exact-time survival/risk from discrete hazards
# =============================================================================
def _clip_hazards_np(h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return np.clip(h, eps, 1.0 - eps)


def hazards_to_survival_end_of_bin_numpy(hazards: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    hazards = _clip_hazards_np(hazards, eps=eps)
    return np.cumprod(1.0 - hazards, axis=1)


def survival_at_time_from_hazards(
    hazards: np.ndarray,
    t_days: float,
    time_bin_width: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Exact-time survival S(t) using within-bin exponential interpolation.
    """
    hazards = _clip_hazards_np(hazards, eps=eps)
    N, K = hazards.shape
    bw = float(time_bin_width)
    bw = max(bw, 1e-6)

    t = float(t_days)
    if t <= 0.0 or K == 0:
        return np.ones((N,), dtype=float)

    k = int(np.floor(t / bw))

    # beyond coverage -> return last end-of-bin survival
    if k >= K:
        S_end = hazards_to_survival_end_of_bin_numpy(hazards, eps=eps)
        return S_end[:, -1]

    log1m = np.log1p(-hazards)  # (N,K) <= 0

    if k == 0:
        frac = t / bw
        return np.exp(log1m[:, 0] * frac)

    logS_prev = np.sum(log1m[:, :k], axis=1)  # end of bin k-1
    within = t - (k * bw)
    lam_k = -log1m[:, k] / bw
    logS_t = logS_prev - lam_k * within
    return np.exp(logS_t)


def risk_at_time_from_hazards(
    hazards: np.ndarray,
    t_days: float,
    time_bin_width: float,
    eps: float = 1e-12,
) -> np.ndarray:
    S = survival_at_time_from_hazards(hazards, t_days, time_bin_width, eps=eps)
    return np.clip(1.0 - S, 0.0, 1.0)


# =============================================================================
# Metrics: C-index (O(n log n)) + KM censoring survival (O(n log n))
# =============================================================================
class _Fenwick:
    def __init__(self, n: int):
        self.n = int(n)
        self.bit = np.zeros(self.n + 1, dtype=np.int64)

    def add(self, i: int, delta: int):
        i = int(i)
        while i <= self.n:
            self.bit[i] += int(delta)
            i += i & -i

    def sum(self, i: int) -> int:
        i = int(i)
        s = 0
        while i > 0:
            s += int(self.bit[i])
            i -= i & -i
        return int(s)

    def total(self) -> int:
        return self.sum(self.n)


def concordance_index(times, events, risks) -> float:
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=float).reshape(-1)
    risks = np.asarray(risks, dtype=float).reshape(-1)

    m = np.isfinite(times) & np.isfinite(events) & np.isfinite(risks)
    times, events, risks = times[m], events[m], risks[m]
    n = times.size
    if n <= 1:
        return 0.0

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    events = events[order]
    risks = risks[order]

    uniq = np.unique(risks)
    rrank = np.searchsorted(uniq, risks, side="left") + 1

    bit = _Fenwick(len(uniq))
    for r in rrank:
        bit.add(r, 1)

    num = 0.0
    den = 0.0

    i = 0
    while i < n:
        t = times[i]
        j = i
        while j < n and times[j] == t:
            j += 1

        for k in range(i, j):
            bit.add(rrank[k], -1)

        total_future = bit.total()
        if total_future > 0:
            for k in range(i, j):
                if events[k] != 1.0:
                    continue
                rk = int(rrank[k])
                lower = bit.sum(rk - 1)
                equal = bit.sum(rk) - bit.sum(rk - 1)
                num += float(lower) + 0.5 * float(equal)
                den += float(total_future)

        i = j

    return float(num / den) if den > 0 else 0.0


def _km_censoring_survival(times: np.ndarray, events: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=float).reshape(-1)
    eval_times = np.asarray(eval_times, dtype=float).reshape(-1)

    m = np.isfinite(times) & np.isfinite(events)
    times = times[m]
    events = events[m]
    if times.size == 0:
        return np.ones_like(eval_times, dtype=float)

    cens = (1.0 - events).astype(int)

    order = np.argsort(times, kind="mergesort")
    t_ord = times[order]
    c_ord = cens[order]

    uniq_t, idx_start = np.unique(t_ord, return_index=True)
    d_cens = np.add.reduceat(c_ord, idx_start)
    n = t_ord.size
    n_at_risk = n - idx_start

    haz = np.zeros_like(d_cens, dtype=float)
    nz = n_at_risk > 0
    haz[nz] = d_cens[nz] / n_at_risk[nz]
    haz = np.clip(haz, 0.0, 1.0)

    G_steps = np.cumprod(1.0 - haz)

    out = np.ones_like(eval_times, dtype=float)
    for i, te in enumerate(eval_times):
        j = np.searchsorted(uniq_t, te, side="right") - 1
        out[i] = max(float(G_steps[j]), 1e-6) if j >= 0 else 1.0
    return out


def integrated_brier_score(times: np.ndarray, events: np.ndarray, hazards: np.ndarray, time_bin_width: float) -> float:
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=float).reshape(-1)
    hazards = np.asarray(hazards, dtype=float)

    if hazards.ndim != 2:
        raise ValueError(f"hazards must be (N,K), got {hazards.shape}")
    N, K = hazards.shape
    if N == 0 or K == 0:
        return 0.0

    bw = float(time_bin_width)
    bw = max(bw, 1e-6)

    S_end = hazards_to_survival_end_of_bin_numpy(hazards)  # (N,K)
    t_grid = bw * (np.arange(K, dtype=float) + 1.0)

    G_t = _km_censoring_survival(times, events, t_grid)
    G_T = _km_censoring_survival(times, events, times)
    G_t = np.clip(G_t, 1e-6, 1.0)
    G_T = np.clip(G_T, 1e-6, 1.0)

    vals: List[float] = []
    for k, t in enumerate(t_grid):
        y = (times > t).astype(float)
        s_pred = S_end[:, k]

        w = np.zeros(N, dtype=float)
        w[times > t] = 1.0 / G_t[k]
        case = (events == 1.0) & (times <= t)
        w[case] = 1.0 / G_T[case]

        vals.append(float(np.sum(w * (y - s_pred) ** 2) / float(N)))

    return float(np.mean(vals)) if vals else 0.0


def time_dependent_auc_surv(
    times: np.ndarray,
    events: np.ndarray,
    hazards: np.ndarray,
    eval_times_days: Sequence[float],
    time_bin_width: float,
) -> Tuple[Dict[float, float], float]:
    """
    IPCW time-dependent AUC using exact-time risk r(t) = 1 - S(t).
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=float).reshape(-1)
    hazards = np.asarray(hazards, dtype=float)

    if hazards.ndim != 2:
        raise ValueError(f"hazards must be (N,K), got {hazards.shape}")
    N, K = hazards.shape
    if N == 0 or K == 0 or len(eval_times_days) == 0:
        return {}, float("nan")

    bw = float(time_bin_width)
    bw = max(bw, 1e-6)

    G_T = _km_censoring_survival(times, events, times)
    G_T = np.clip(G_T, 1e-6, 1.0)

    auc_by_time: Dict[float, float] = {}
    auc_vals: List[float] = []

    for t in eval_times_days:
        t = float(t)
        score = risk_at_time_from_hazards(hazards, t, bw)

        cases = (events == 1.0) & (times <= t)
        ctrls = (times > t)
        if int(cases.sum()) == 0 or int(ctrls.sum()) == 0:
            auc_by_time[t] = float("nan")
            continue

        G_t = float(np.clip(_km_censoring_survival(times, events, np.array([t], dtype=float))[0], 1e-6, 1.0))

        s_case = score[cases]
        s_ctrl = score[ctrls]
        w_case = 1.0 / G_T[cases]
        w_ctrl = np.full(s_ctrl.shape, 1.0 / G_t, dtype=float)

        diff = s_case[:, None] - s_ctrl[None, :]
        w_pairs = w_case[:, None] * w_ctrl[None, :]
        den = float(np.sum(w_pairs))
        if den <= 0.0:
            auc_by_time[t] = float("nan")
            continue

        num = float(np.sum(w_pairs * (diff > 0.0)) + 0.5 * np.sum(w_pairs * (diff == 0.0)))
        auc_t = num / den
        auc_by_time[t] = float(auc_t)
        auc_vals.append(float(auc_t))

    return auc_by_time, (float(np.mean(auc_vals)) if auc_vals else float("nan"))


def decision_curve_analysis_surv(
    times: np.ndarray,
    events: np.ndarray,
    hazards: np.ndarray,
    eval_times_days: Sequence[float],
    thresholds: Sequence[float],
    time_bin_width: float,
) -> Tuple[Dict[Tuple[float, float], float], float]:
    """
    IPCW Decision Curve Analysis using exact-time risk r(t).
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=float).reshape(-1)
    hazards = np.asarray(hazards, dtype=float)

    if hazards.ndim != 2:
        raise ValueError(f"hazards must be (N,K), got {hazards.shape}")
    N, K = hazards.shape
    if N == 0 or K == 0 or len(eval_times_days) == 0 or len(thresholds) == 0:
        return {}, float("nan")

    bw = float(time_bin_width)
    bw = max(bw, 1e-6)

    G_T = _km_censoring_survival(times, events, times)
    G_T = np.clip(G_T, 1e-6, 1.0)

    nb_by: Dict[Tuple[float, float], float] = {}
    nb_vals: List[float] = []

    thresholds = [float(c) for c in thresholds]

    for t in eval_times_days:
        t = float(t)
        score = risk_at_time_from_hazards(hazards, t, bw)

        y_case = ((events == 1.0) & (times <= t)).astype(float)
        G_t = float(np.clip(_km_censoring_survival(times, events, np.array([t], dtype=float))[0], 1e-6, 1.0))

        w = np.zeros(N, dtype=float)
        w[times > t] = 1.0 / G_t
        case = (events == 1.0) & (times <= t)
        w[case] = 1.0 / G_T[case]

        for c in thresholds:
            if not (0.0 < c < 1.0):
                nb_by[(t, c)] = float("nan")
                continue

            treat = (score >= c).astype(float)
            TP = float(np.sum(w * treat * (y_case == 1.0)) / float(N))
            FP = float(np.sum(w * treat * (y_case == 0.0)) / float(N))
            nb = TP - FP * (c / (1.0 - c))

            nb_by[(t, c)] = float(nb)
            nb_vals.append(float(nb))

    return nb_by, (float(np.mean(nb_vals)) if nb_vals else float("nan"))


def ipcw_auc_at_horizon_from_risk(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    horizon_days: float,
) -> float:
    times = np.asarray(times, dtype=float).reshape(-1)
    events = np.asarray(events, dtype=float).reshape(-1)
    risk_scores = np.asarray(risk_scores, dtype=float).reshape(-1)

    m = np.isfinite(times) & np.isfinite(events) & np.isfinite(risk_scores)
    times, events, risk_scores = times[m], events[m], risk_scores[m]
    if times.size == 0:
        return float("nan")

    t = float(horizon_days)
    cases = (events == 1.0) & (times <= t)
    ctrls = (times > t)
    if int(cases.sum()) == 0 or int(ctrls.sum()) == 0:
        return float("nan")

    G_T = _km_censoring_survival(times, events, times)
    G_T = np.clip(G_T, 1e-6, 1.0)
    G_t = float(np.clip(_km_censoring_survival(times, events, np.array([t], dtype=float))[0], 1e-6, 1.0))

    s_case = risk_scores[cases]
    s_ctrl = risk_scores[ctrls]
    w_case = 1.0 / G_T[cases]
    w_ctrl = np.full(s_ctrl.shape, 1.0 / G_t, dtype=float)

    diff = s_case[:, None] - s_ctrl[None, :]
    w_pairs = w_case[:, None] * w_ctrl[None, :]
    den = float(np.sum(w_pairs))
    if den <= 0.0:
        return float("nan")

    num = float(np.sum(w_pairs * (diff > 0.0)) + 0.5 * np.sum(w_pairs * (diff == 0.0)))
    return float(num / den)
