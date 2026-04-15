"""Shared radiomics feature encoder for TriFuseSurv."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def _pad_or_trunc_1d(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] == dim:
        return x
    if x.shape[0] > dim:
        return x[:dim].copy()
    out = np.zeros(dim, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


class RadiomicsEncoder:
    """PCA-based grouped radiomics encoder.

    Supports either:
    - a directory of per-patient radiomics CSVs, or
    - a single patient-wide CSV with grouped columns such as
      `PT_intratumor__...`, `PT_peritumor_10mm__...`,
      `LN_intratumor__...`, `LN_peritumor_10mm__...`.
    """

    def __init__(
        self,
        patient_vectors: Dict[str, np.ndarray],
        *,
        group_names: Optional[List[str]] = None,
        group_n_comp: Optional[Dict[str, int]] = None,
        pc_slices: Optional[Dict[str, slice]] = None,
        total_pc_dim: int = 0,
    ):
        self.patient_vectors: Dict[str, np.ndarray] = {}
        for k, v in patient_vectors.items():
            vec = np.asarray(v, dtype=np.float32).reshape(-1)
            raw_key = str(k)
            norm_key = self.normalize_patient_id(raw_key)
            self.patient_vectors[raw_key] = vec
            self.patient_vectors.setdefault(norm_key, vec)
        if not self.patient_vectors:
            raise RuntimeError("[RAD] empty patient_vectors")
        first = next(iter(self.patient_vectors.values()))
        self.output_dim = int(first.shape[0])
        self.group_names = list(group_names or [])
        self.group_n_comp = dict(group_n_comp or {})
        self.pc_slices = dict(pc_slices or {})
        self.total_pc_dim = int(total_pc_dim)

    @staticmethod
    def normalize_patient_id(pid: Any) -> str:
        s = str(pid).strip()
        s = re.sub(r"(_radio|_radiomics|_rad)$", "", s, flags=re.IGNORECASE)
        m = re.match(r"^([A-Za-z]+)0*([0-9]+)$", s)
        if m:
            return f"{m.group(1).upper()}{int(m.group(2))}"
        return s.upper()

    @staticmethod
    def build_radiomics_path(lid: str, radiomics_root: str) -> Optional[Path]:
        lid = str(lid)
        roots = [Path(radiomics_root), Path("radiomics") / radiomics_root]
        basenames = [f"{lid}_radio_radiomics.csv", f"{lid}_radiomics.csv", f"{lid}.csv"]
        for root in roots:
            for bn in basenames:
                p = root / bn
                if p.is_file():
                    return p
        return None

    @staticmethod
    def _extract_feature_cols(df: pd.DataFrame) -> List[str]:
        meta_cols = {
            "roi_name",
            "case_id",
            "tumor_id",
            "tumor_class",
            "tumor_origin_roi",
            "region",
            "peritumor_radius_mm",
            "qc_voxel_volume_mm3",
            "qc_roi_voxels",
            "qc_roi_volume_ml",
        }
        return sorted([c for c in df.columns if (not c.startswith("diagnostics_")) and c not in meta_cols])

    @staticmethod
    def _aggregate_patient_groups(df: pd.DataFrame, feature_cols: Optional[List[str]]):
        if feature_cols is None:
            feature_cols = RadiomicsEncoder._extract_feature_cols(df)
        else:
            feature_cols = list(feature_cols)
            for c in feature_cols:
                if c not in df.columns:
                    df[c] = np.nan

        df_feat = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        idx = df.index
        roi = (
            df["roi_name"].astype(str)
            if "roi_name" in df.columns
            else pd.Series(["PT_intratumor"] * len(df), index=idx)
        )
        tumor_class = (
            df["tumor_class"].astype(str)
            if "tumor_class" in df.columns
            else roi.map(lambda s: "PT" if str(s).upper().startswith("PT") else "LN")
        )
        region = (
            df["region"].astype(str)
            if "region" in df.columns
            else roi.map(lambda s: "peritumor" if "peri" in str(s).lower() else "intratumor")
        )

        tumor_class_u = tumor_class.str.upper()
        region_l = region.str.lower()
        is_pt = tumor_class_u == "PT"
        is_ln = ~is_pt

        group_specs = [
            ("PT_intra", is_pt & (region_l == "intratumor")),
            ("PT_peri", is_pt & (region_l == "peritumor")),
            ("LN_intra", is_ln & (region_l == "intratumor")),
            ("LN_peri", is_ln & (region_l == "peritumor")),
        ]

        n_feat = len(feature_cols)
        group_vectors = {}
        presence_bits = []
        for gname, mask in group_specs:
            if mask.any():
                sub = df_feat.loc[mask]
                mean = sub.mean(axis=0).to_numpy(dtype=np.float32)
                std = sub.std(axis=0).to_numpy(dtype=np.float32)
                presence_bits.append(1.0)
            else:
                mean = np.zeros(n_feat, dtype=np.float32)
                std = np.zeros(n_feat, dtype=np.float32)
                presence_bits.append(0.0)
            mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
            std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
            group_vectors[gname] = np.concatenate([mean, std], axis=0)

        presence = np.asarray(presence_bits, dtype=np.float32)
        group_names = [g[0] for g in group_specs]
        return group_vectors, presence, feature_cols, group_names

    @classmethod
    def _fit_from_group_maps(
        cls,
        *,
        train_ids: Sequence[str],
        all_ids: Sequence[str],
        group_names: List[str],
        group_vectors: Dict[str, Dict[str, np.ndarray]],
        presence_map: Dict[str, np.ndarray],
        total_pcs: int,
        seed: int,
    ) -> "RadiomicsEncoder":
        available_ids = [lid for lid in all_ids if lid in presence_map]
        if not available_ids:
            raise RuntimeError("[RAD] No usable radiomics rows matched requested IDs.")

        G = len(group_names)
        group_index = {g: i for i, g in enumerate(group_names)}
        pcs_per_group_target = max(1, int(total_pcs) // max(1, G))

        group_pca_means: Dict[str, Optional[np.ndarray]] = {}
        group_pca_components: Dict[str, Optional[np.ndarray]] = {}
        group_n_comp: Dict[str, int] = {}

        for g in group_names:
            rows = []
            for lid in train_ids:
                pres = presence_map.get(lid)
                if pres is None or pres[group_index[g]] < 0.5:
                    continue
                vec = group_vectors[g].get(lid)
                if vec is not None and vec.size > 0:
                    rows.append(vec)
            if not rows:
                group_pca_means[g] = None
                group_pca_components[g] = None
                group_n_comp[g] = 0
                continue
            X = np.stack(rows, axis=0).astype(np.float32)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            n_samples, dim = X.shape
            n_comp = max(1, min(pcs_per_group_target, n_samples, dim))
            pca = PCA(n_components=n_comp, svd_solver="full", random_state=int(seed))
            pca.fit(X)
            group_pca_means[g] = pca.mean_.astype(np.float32)
            group_pca_components[g] = pca.components_.astype(np.float32)
            group_n_comp[g] = int(n_comp)

        total_pc_dim = sum(group_n_comp[g] for g in group_names)
        final_dim = total_pc_dim + G
        pc_slices: Dict[str, slice] = {}
        offset = 0
        for g in group_names:
            nc = group_n_comp[g]
            if nc > 0:
                pc_slices[g] = slice(offset, offset + nc)
                offset += nc

        patient_vectors = {}
        for lid in available_ids:
            pcs_chunks = []
            pres_vec = presence_map[lid]
            for g in group_names:
                n_comp = group_n_comp[g]
                mean = group_pca_means[g]
                comp = group_pca_components[g]
                if n_comp <= 0 or mean is None or comp is None:
                    continue
                if pres_vec[group_index[g]] < 0.5:
                    pcs = np.zeros((n_comp,), dtype=np.float32)
                else:
                    x_raw = group_vectors[g].get(lid)
                    if x_raw is None:
                        pcs = np.zeros((n_comp,), dtype=np.float32)
                    else:
                        x_raw = np.nan_to_num(np.asarray(x_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                        pcs = (comp @ (x_raw - mean)).astype(np.float32)
                pcs_chunks.append(pcs)

            pcs_all = np.concatenate(pcs_chunks, axis=0) if pcs_chunks else np.zeros((total_pc_dim,), dtype=np.float32)
            pcs_all = _pad_or_trunc_1d(pcs_all, total_pc_dim)
            presence = _pad_or_trunc_1d(pres_vec.astype(np.float32), G)
            patient_vectors[lid] = _pad_or_trunc_1d(np.concatenate([pcs_all, presence], axis=0), final_dim)

        return cls(
            patient_vectors,
            group_names=group_names,
            group_n_comp=group_n_comp,
            pc_slices=pc_slices,
            total_pc_dim=total_pc_dim,
        )

    @classmethod
    def fit(cls, train_ids: Sequence[str], all_ids: Sequence[str], radiomics_root: str, total_pcs: int, seed: int) -> "RadiomicsEncoder":
        source_path = Path(radiomics_root)
        if source_path.is_file():
            return cls.fit_from_wide_csv(train_ids, all_ids, source_path, total_pcs, seed)
        return cls.fit_from_directory(train_ids, all_ids, radiomics_root, total_pcs, seed)

    @classmethod
    def fit_from_directory(
        cls,
        train_ids: Sequence[str],
        all_ids: Sequence[str],
        radiomics_root: str,
        total_pcs: int,
        seed: int,
    ) -> "RadiomicsEncoder":
        all_ids = list(dict.fromkeys([str(x) for x in all_ids]))
        train_ids = [str(x) for x in train_ids]

        feature_cols = None
        for lid in train_ids:
            p = cls.build_radiomics_path(lid, radiomics_root)
            if p is None:
                continue
            try:
                df0 = pd.read_csv(p)
            except Exception:
                continue
            feature_cols = cls._extract_feature_cols(df0)
            if feature_cols:
                break

        if feature_cols is None:
            raise RuntimeError("[RAD] Could not determine feature columns from TRAIN radiomics CSVs.")

        group_names = None
        group_vectors: Dict[str, Dict[str, np.ndarray]] = {}
        presence_map: Dict[str, np.ndarray] = {}

        for lid in all_ids:
            p = cls.build_radiomics_path(lid, radiomics_root)
            if p is None:
                continue
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            gvecs, presence, _, gnames = cls._aggregate_patient_groups(df, feature_cols)
            lid_key = cls.normalize_patient_id(lid)
            if group_names is None:
                group_names = gnames
                group_vectors = {g: {} for g in group_names}
            for g, vec in gvecs.items():
                group_vectors[g][lid_key] = vec.astype(np.float32)
            presence_map[lid_key] = presence.astype(np.float32)

        if group_names is None or not presence_map:
            raise RuntimeError("[RAD] No usable radiomics CSVs found.")

        return cls._fit_from_group_maps(
            train_ids=[cls.normalize_patient_id(x) for x in train_ids],
            all_ids=[cls.normalize_patient_id(x) for x in all_ids],
            group_names=group_names,
            group_vectors=group_vectors,
            presence_map=presence_map,
            total_pcs=total_pcs,
            seed=seed,
        )

    @classmethod
    def fit_from_wide_csv(
        cls,
        train_ids: Sequence[str],
        all_ids: Sequence[str],
        radiomics_csv: str | Path,
        total_pcs: int,
        seed: int,
    ) -> "RadiomicsEncoder":
        train_ids = [cls.normalize_patient_id(x) for x in train_ids]
        all_ids = list(dict.fromkeys(cls.normalize_patient_id(x) for x in all_ids))
        id_set = set(all_ids)

        df = pd.read_csv(radiomics_csv)
        if "case_id" not in df.columns:
            raise ValueError(f"[RAD] Radiomics CSV missing required 'case_id' column: {radiomics_csv}")

        df = df.copy()
        df["patient_id_norm"] = df["case_id"].map(cls.normalize_patient_id)
        dup_mask = df["patient_id_norm"].duplicated(keep=False)
        if dup_mask.any():
            dup_ids = df.loc[dup_mask, "patient_id_norm"].drop_duplicates().tolist()
            raise ValueError(f"[RAD] duplicate normalized radiomics IDs in {radiomics_csv}: {dup_ids[:10]}")

        group_specs = {
            "PT_intra": "PT_intratumor__",
            "PT_peri": "PT_peritumor_10mm__",
            "LN_intra": "LN_intratumor__",
            "LN_peri": "LN_peritumor_10mm__",
        }
        presence_cols = {
            "PT_intra": "present__PT_intratumor",
            "PT_peri": "present__PT_peritumor_10mm",
            "LN_intra": "present__LN_intratumor",
            "LN_peri": "present__LN_peritumor_10mm",
        }
        group_names = ["PT_intra", "PT_peri", "LN_intra", "LN_peri"]
        group_feature_cols: Dict[str, List[str]] = {}

        for group_name, prefix in group_specs.items():
            cols = [c for c in df.columns if c.startswith(prefix) and not c.endswith("__error")]
            numeric_cols = []
            for col in cols:
                ser = pd.to_numeric(df[col], errors="coerce")
                if ser.notna().any():
                    numeric_cols.append(col)
            group_feature_cols[group_name] = sorted(numeric_cols)

        if not any(group_feature_cols.values()):
            raise RuntimeError(f"[RAD] No usable grouped radiomics columns found in {radiomics_csv}")

        group_vectors: Dict[str, Dict[str, np.ndarray]] = {g: {} for g in group_names}
        presence_map: Dict[str, np.ndarray] = {}
        for _, row in df.iterrows():
            pid = str(row["patient_id_norm"])
            if pid not in id_set:
                continue

            presence_bits = []
            for group_name in group_names:
                cols = group_feature_cols[group_name]
                if cols:
                    vals = pd.to_numeric(pd.Series([row[c] for c in cols]), errors="coerce").to_numpy(dtype=np.float32)
                else:
                    vals = np.zeros(0, dtype=np.float32)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                group_vectors[group_name][pid] = vals

                pcol = presence_cols[group_name]
                if pcol in df.columns:
                    try:
                        present = float(row[pcol]) > 0.0
                    except Exception:
                        present = bool(np.any(np.abs(vals) > 0))
                else:
                    present = bool(np.any(np.abs(vals) > 0))
                presence_bits.append(1.0 if present else 0.0)

            presence_map[pid] = np.asarray(presence_bits, dtype=np.float32)

        if not presence_map:
            raise RuntimeError(f"[RAD] No usable radiomics rows matched requested IDs in {radiomics_csv}")

        return cls._fit_from_group_maps(
            train_ids=train_ids,
            all_ids=all_ids,
            group_names=group_names,
            group_vectors=group_vectors,
            presence_map=presence_map,
            total_pcs=total_pcs,
            seed=seed,
        )

    def encode_patient(self, lid: str) -> np.ndarray:
        raw_key = str(lid)
        vec = self.patient_vectors.get(raw_key)
        if vec is None:
            vec = self.patient_vectors.get(self.normalize_patient_id(raw_key))
        if vec is None:
            return np.zeros(self.output_dim, dtype=np.float32)
        return _pad_or_trunc_1d(vec, self.output_dim)
