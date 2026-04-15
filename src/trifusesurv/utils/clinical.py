"""Shared clinical feature encoder for TriFuseSurv."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# Schema: explicit type for known clinical columns
CLINICAL_SCHEMA: Dict[str, str] = {
    "AGE": "numeric",
    "KFCF": "numeric",
    "T": "ordinal",
    "N": "ordinal",
    "M": "ordinal",
    "NSTAGE": "ordinal",
    "SMOKE": "ordinal",
    "ALCOHOL": "ordinal",
    "HPV": "ordinal",
}

DEFAULT_CLINICAL_COLS = [
    "PATHOLOGY", "NSTAGE", "AGE", "SEX", "RACE",
    "T", "N", "M", "KFCF", "TX", "SMOKE", "ALCOHOL", "HPV",
]

ENDPOINT_MAP = {
    "OS": ("OS.TIME", "OS.EVENT"),
    "DSS": ("DSS.TIME", "DSS.EVENT"),
    "DFS": ("DFS.TIME", "DFS.EVENT"),
}


def parse_ordinal_value(col: str, val: Any) -> float:
    """Parse an ordinal/numeric clinical value to float."""
    if val is None or pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s == "":
        return np.nan
    if col.upper() in {"NSTAGE", "STAGE"}:
        s = s.replace("STAGE", "").replace("STG", "").strip()
    m = re.search(r"(-?\d+(\.\d+)?)", s)
    if m is not None:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    roman_map = {"0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}
    if s in roman_map:
        return float(roman_map[s])
    if s in {"YES", "Y", "POS", "POSITIVE"}:
        return 1.0
    if s in {"NO", "N", "NEG", "NEGATIVE"}:
        return 0.0
    if col.upper() in {"T", "N", "M"} and "IS" in s:
        return 0.0
    return np.nan


class ClinicalEncoder:
    """One-hot categorical + z-scored numeric clinical feature encoder."""

    def __init__(
        self,
        numeric_cols: List[str],
        numeric_means: Dict[str, float],
        numeric_stds: Dict[str, float],
        cat_cols: List[str],
        cat_maps: Dict[str, Dict[str, int]],
        cat_dims: Dict[str, int],
    ):
        self.numeric_cols = numeric_cols
        self.numeric_means = numeric_means
        self.numeric_stds = numeric_stds
        self.cat_cols = cat_cols
        self.cat_maps = cat_maps
        self.cat_dims = cat_dims
        self.output_dim = len(self.numeric_cols) * 2 + sum(self.cat_dims[c] for c in self.cat_cols)

    @staticmethod
    def _is_numeric_column(series: pd.Series, threshold: float = 0.8) -> bool:
        coerced = pd.to_numeric(series, errors="coerce")
        valid = coerced.notna().sum()
        frac = valid / max(len(series), 1)
        return frac >= threshold

    @classmethod
    def fit(cls, df: pd.DataFrame, clinical_cols: List[str]) -> "ClinicalEncoder":
        numeric_cols: List[str] = []
        numeric_means: Dict[str, float] = {}
        numeric_stds: Dict[str, float] = {}
        cat_cols: List[str] = []
        cat_maps: Dict[str, Dict[str, int]] = {}
        cat_dims: Dict[str, int] = {}

        for col in clinical_cols:
            if col not in df.columns:
                print(f"[CLIN] Column {col} not found; skipping.")
                continue
            series = df[col]
            non_na = series.dropna()
            if len(non_na) == 0:
                print(f"[CLIN] Column {col} all NaNs; skipping.")
                continue

            schema = CLINICAL_SCHEMA.get(col, "auto")
            if schema in ("numeric", "ordinal"):
                treat_as_numeric = True
            elif schema == "categorical":
                treat_as_numeric = False
            else:
                treat_as_numeric = cls._is_numeric_column(series)

            if treat_as_numeric:
                if schema == "ordinal":
                    codes = series.apply(lambda v, c=col: parse_ordinal_value(c, v))
                    series_num = pd.to_numeric(codes, errors="coerce")
                else:
                    series_num = pd.to_numeric(series, errors="coerce")
                valid = series_num.notna()
                if valid.sum() == 0:
                    print(f"[CLIN] Column {col} has no valid numeric; skipping.")
                    continue
                mu = float(series_num[valid].mean())
                sd = float(series_num[valid].std())
                sd = sd if sd > 1e-6 else 1.0
                numeric_cols.append(col)
                numeric_means[col] = mu
                numeric_stds[col] = sd
            else:
                cats = sorted({str(v).strip() for v in non_na.values if str(v).strip() != ""})
                if not cats:
                    print(f"[CLIN] Column {col} has no categories; skipping.")
                    continue
                mapping = {cat: idx for idx, cat in enumerate(cats)}
                cat_cols.append(col)
                cat_maps[col] = mapping
                cat_dims[col] = len(cats) + 1  # +UNK

        enc = cls(numeric_cols, numeric_means, numeric_stds, cat_cols, cat_maps, cat_dims)
        print(f"[CLIN] Clinical dim={enc.output_dim}")
        return enc

    def encode_row(self, row: pd.Series) -> np.ndarray:
        feats: List[float] = []

        for col in self.numeric_cols:
            val = row.get(col, np.nan)
            schema = CLINICAL_SCHEMA.get(col, "auto")
            x = np.nan
            if val is not None and not pd.isna(val):
                try:
                    x = parse_ordinal_value(col, val) if schema == "ordinal" else float(val)
                except Exception:
                    x = np.nan
            if np.isnan(x):
                feats.append(0.0)
                feats.append(1.0)
            else:
                z = (x - self.numeric_means[col]) / self.numeric_stds[col]
                feats.append(float(z))
                feats.append(0.0)

        for col in self.cat_cols:
            dim = self.cat_dims[col]
            one_hot = np.zeros(dim, dtype=np.float32)
            val = row.get(col, None)
            unk = dim - 1
            if val is not None and not pd.isna(val):
                idx = self.cat_maps[col].get(str(val).strip(), unk)
            else:
                idx = unk
            one_hot[idx] = 1.0
            feats.extend(one_hot.tolist())

        return np.asarray(feats, dtype=np.float32) if feats else np.zeros(0, dtype=np.float32)

    def feature_groups(self) -> Dict[str, List[int]]:
        """Map each clinical column to its output indices."""
        groups: Dict[str, List[int]] = {}
        idx = 0
        for col in self.numeric_cols:
            groups[col] = [idx, idx + 1]
            idx += 2
        for col in self.cat_cols:
            dim = int(self.cat_dims[col])
            groups[col] = list(range(idx, idx + dim))
            idx += dim
        return groups


class ClinicalEncoderCompact:
    """Compact clinical encoder that matches checkpoint dimensions exactly.

    Uses ordinal encoding (z-scored) for categoricals instead of one-hot,
    with a target_dim parameter to auto-adjust dims to match a checkpoint.
    Used by SHAP export for deterministic dim matching.
    """

    def __init__(self, specs: List[Dict[str, Any]]):
        self.specs = list(specs)
        self.output_dim = int(sum(2 if s["kind"].endswith("2") else 1 for s in self.specs))

    @classmethod
    def fit(
        cls,
        df_train: pd.DataFrame,
        clinical_cols: List[str],
        *,
        global_cat_maps: Dict[str, Dict[str, int]],
        target_dim: int = 0,
    ) -> "ClinicalEncoderCompact":
        clinical_cols = [c for c in list(clinical_cols) if c in df_train.columns]

        specs: List[Dict[str, Any]] = []
        for col in clinical_cols:
            schema = CLINICAL_SCHEMA.get(col, "auto")
            s = df_train[col]

            if schema in ("numeric", "ordinal"):
                vals = []
                for v in s.values:
                    if v is None or pd.isna(v):
                        vals.append(np.nan)
                    elif schema == "ordinal":
                        vals.append(parse_ordinal_value(col, v))
                    else:
                        try:
                            vals.append(float(v))
                        except Exception:
                            vals.append(np.nan)
                arr = np.asarray(vals, dtype=np.float32)
                miss = float(np.mean(~np.isfinite(arr))) if arr.size > 0 else 0.0
                ok = np.isfinite(arr)
                mu = float(arr[ok].mean()) if ok.any() else 0.0
                sd = float(arr[ok].std()) if ok.any() else 1.0
                if not np.isfinite(sd) or sd < 1e-6:
                    sd = 1.0
                specs.append(dict(col=col, kind="num2", mean=mu, std=sd, cat_map=None, miss_rate=miss))
            else:
                mp = global_cat_maps.get(col, {})
                codes = []
                for v in s.values:
                    if v is None or pd.isna(v):
                        codes.append(np.nan)
                    else:
                        key = str(v).strip()
                        codes.append(float(mp[key]) if (key != "" and key in mp) else np.nan)
                arr = np.asarray(codes, dtype=np.float32)
                miss = float(np.mean(~np.isfinite(arr))) if arr.size > 0 else 0.0
                ok = np.isfinite(arr)
                mu = float(arr[ok].mean()) if ok.any() else 0.0
                sd = float(arr[ok].std()) if ok.any() else 1.0
                if not np.isfinite(sd) or sd < 1e-6:
                    sd = 1.0
                specs.append(dict(col=col, kind="cat1", mean=mu, std=sd, cat_map=mp, miss_rate=miss))

        def cur_dim(ss):
            return int(sum(2 if s["kind"].endswith("2") else 1 for s in ss))

        if int(target_dim) > 0:
            d = cur_dim(specs)
            if d > int(target_dim):
                cand = [s for s in specs if s["kind"] in ("num2", "cat2")]
                cand.sort(key=lambda s: (s["miss_rate"], s["col"]))
                i = 0
                while d > int(target_dim) and i < len(cand):
                    s = cand[i]
                    if s["kind"] == "num2":
                        s["kind"] = "num1"
                        d -= 1
                    elif s["kind"] == "cat2":
                        s["kind"] = "cat1"
                        d -= 1
                    i += 1
            if d < int(target_dim):
                cand = [s for s in specs if s["kind"] in ("num1", "cat1")]
                cand.sort(key=lambda s: (-s["miss_rate"], s["col"]))
                i = 0
                while d < int(target_dim) and i < len(cand):
                    s = cand[i]
                    if s["kind"] == "num1":
                        s["kind"] = "num2"
                        d += 1
                    elif s["kind"] == "cat1":
                        s["kind"] = "cat2"
                        d += 1
                    i += 1
            if d != int(target_dim):
                raise RuntimeError(f"[CLIN] Could not match ckpt clinical_dim={target_dim}. Got dim={d}.")

        enc = cls(specs)
        print(f"[CLIN] Clinical dim={enc.output_dim} (target={int(target_dim) if int(target_dim) > 0 else 'auto'})")
        return enc

    def encode_row(self, row: pd.Series) -> np.ndarray:
        feats: List[float] = []
        for s in self.specs:
            schema = CLINICAL_SCHEMA.get(s["col"], "auto")
            v = row.get(s["col"], None)

            if s["kind"].startswith("num"):
                x = np.nan
                if v is not None and not pd.isna(v):
                    try:
                        x = parse_ordinal_value(s["col"], v) if schema == "ordinal" else float(v)
                    except Exception:
                        x = np.nan
                if not np.isfinite(x):
                    z, miss = 0.0, 1.0
                else:
                    z, miss = float((float(x) - float(s["mean"])) / float(s["std"])), 0.0
                feats.append(z)
                if s["kind"].endswith("2"):
                    feats.append(miss)
            else:
                mp = s["cat_map"] or {}
                if v is None or pd.isna(v):
                    code = np.nan
                else:
                    key = str(v).strip()
                    code = float(mp[key]) if (key != "" and key in mp) else np.nan
                if not np.isfinite(code):
                    z, miss = 0.0, 1.0
                else:
                    z, miss = float((float(code) - float(s["mean"])) / float(s["std"])), 0.0
                feats.append(z)
                if s["kind"].endswith("2"):
                    feats.append(miss)

        return np.asarray(feats, dtype=np.float32) if feats else np.zeros((0,), dtype=np.float32)

    def feature_groups(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {}
        idx = 0
        for s in self.specs:
            d = 2 if s["kind"].endswith("2") else 1
            groups[s["col"]] = list(range(idx, idx + d))
            idx += d
        return groups
