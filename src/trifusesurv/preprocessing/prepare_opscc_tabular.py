#!/usr/bin/env python3
"""Prepare the stage-2 OPSCC metafile from preprocessed imaging plus tabular CSVs."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional, Set

import pandas as pd

from trifusesurv.utils.clinical import DEFAULT_CLINICAL_COLS


SURVIVAL_COLUMNS = ["Patient_ID", "OS.TIME", "OS.EVENT", "DSS.TIME", "DSS.EVENT", "DFS.TIME", "DFS.EVENT"]
CLINICAL_ID_CANDIDATES = ["L_ID", "Patient_ID", "patient_id"]
SURVIVAL_ENDPOINTS = (
    ("OS.TIME", "OS.EVENT"),
    ("DSS.TIME", "DSS.EVENT"),
    ("DFS.TIME", "DFS.EVENT"),
)


def normalize_patient_id(pid: str) -> str:
    s = str(pid).strip()
    s = re.sub(r"(_radio|_radiomics|_rad)$", "", s, flags=re.IGNORECASE)
    m = re.match(r"^([A-Za-z]+)0*([0-9]+)$", s)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return s.upper()


def _ensure_unique(df: pd.DataFrame, key: str, *, context: str) -> pd.DataFrame:
    dup_mask = df[key].duplicated(keep=False)
    if dup_mask.any():
        dup_ids = df.loc[dup_mask, key].astype(str).drop_duplicates().tolist()
        raise ValueError(f"{context} has duplicate normalized IDs: {dup_ids[:10]}")
    return df


def load_survival_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in SURVIVAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Survival CSV missing columns: {missing}")
    df = df.copy()
    df["patient_id_norm"] = df["Patient_ID"].map(normalize_patient_id)
    df = _ensure_unique(df, "patient_id_norm", context=f"survival CSV {path}")
    for col in SURVIVAL_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    has_any_endpoint = pd.Series(False, index=df.index)
    for time_col, event_col in SURVIVAL_ENDPOINTS:
        has_any_endpoint = has_any_endpoint | (df[time_col].notna() & df[event_col].notna())
    return df.loc[has_any_endpoint].copy()


def load_clinical_data(path: str, target_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    id_col = next((col for col in CLINICAL_ID_CANDIDATES if col in df.columns), None)
    if id_col is None:
        raise ValueError(f"Clinical CSV missing ID column; expected one of {CLINICAL_ID_CANDIDATES}")

    df = df.copy().rename(columns={id_col: "clinical_source_id"})
    df["patient_id_norm"] = df["clinical_source_id"].map(normalize_patient_id)
    df = _ensure_unique(df, "patient_id_norm", context=f"clinical CSV {path}")
    if target_ids is not None:
        df = df[df["patient_id_norm"].isin(target_ids)].copy()
    return df


def load_radiomics_data(path: str, target_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "case_id" not in df.columns:
        raise ValueError(f"Radiomics CSV must contain 'case_id' column, got: {df.columns.tolist()}")
    df = df.copy()
    df["patient_id_norm"] = df["case_id"].map(normalize_patient_id)
    df = _ensure_unique(df, "patient_id_norm", context=f"radiomics CSV {path}")
    if target_ids is not None:
        df = df[df["patient_id_norm"].isin(target_ids)].copy()
    return df


def load_base_meta(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "patient_id" not in df.columns:
        raise ValueError(f"Base meta CSV must contain 'patient_id': {path}")
    df = df.copy()
    df["patient_id_norm"] = df["patient_id"].map(normalize_patient_id)
    return _ensure_unique(df, "patient_id_norm", context=f"base meta CSV {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_meta_csv", type=str, default="", help="Preprocessed imaging metafile to augment for stage 2")
    p.add_argument("--surv_csv", type=str, required=True)
    p.add_argument("--clin_csv", type=str, default="")
    p.add_argument("--radio_csv", type=str, default="")
    p.add_argument("--out_csv", type=str, default="cohort_preprocessed_stage2.csv")
    p.add_argument("--out_dir", type=str, default=".")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    surv_df = load_survival_data(args.surv_csv)
    surv_ids = set(surv_df["patient_id_norm"].unique())

    if args.base_meta_csv:
        merged_df = load_base_meta(args.base_meta_csv)
    else:
        merged_df = surv_df[["Patient_ID", "patient_id_norm"]].rename(columns={"Patient_ID": "patient_id"}).copy()
        merged_df["ct_out_path"] = ""
        merged_df["mask_primary_out_path"] = ""
        merged_df["mask_nodal_out_path"] = ""
        merged_df["status"] = "ok"

    merged_df = merged_df.drop(columns=[c for c in SURVIVAL_COLUMNS[1:] if c in merged_df.columns], errors="ignore")
    merged_df = merged_df.merge(
        surv_df[["patient_id_norm", *SURVIVAL_COLUMNS[1:]]],
        on="patient_id_norm",
        how="left",
    )
    merged_df["survival_matched_os"] = merged_df["OS.TIME"].notna() & merged_df["OS.EVENT"].notna()
    merged_df["survival_matched_dss"] = merged_df["DSS.TIME"].notna() & merged_df["DSS.EVENT"].notna()
    merged_df["survival_matched_dfs"] = merged_df["DFS.TIME"].notna() & merged_df["DFS.EVENT"].notna()
    merged_df["survival_matched_any"] = (
        merged_df["survival_matched_os"]
        | merged_df["survival_matched_dss"]
        | merged_df["survival_matched_dfs"]
    )
    merged_df["survival_matched"] = merged_df["survival_matched_any"]

    if args.clin_csv and os.path.exists(args.clin_csv):
        clin_df = load_clinical_data(args.clin_csv, target_ids=surv_ids)
        clinical_value_cols = [c for c in DEFAULT_CLINICAL_COLS if c in clin_df.columns]
        if clinical_value_cols:
            merged_df = merged_df.drop(columns=clinical_value_cols, errors="ignore")
            merged_df = merged_df.merge(
                clin_df[["patient_id_norm", *clinical_value_cols]],
                on="patient_id_norm",
                how="left",
            )
            merged_df["clinical_matched"] = ~merged_df[clinical_value_cols].isna().all(axis=1)
        else:
            merged_df["clinical_matched"] = False
    else:
        merged_df["clinical_matched"] = False

    if args.radio_csv and os.path.exists(args.radio_csv):
        radio_df = load_radiomics_data(args.radio_csv, target_ids=surv_ids)
        radio_ids = set(radio_df["patient_id_norm"].tolist())
        merged_df["radiomics_matched"] = merged_df["patient_id_norm"].isin(radio_ids)
    else:
        merged_df["radiomics_matched"] = False

    out_csv_path = os.path.join(args.out_dir, args.out_csv)
    merged_df.to_csv(out_csv_path, index=False)

    summary = {
        "n_patients": int(len(merged_df)),
        "n_status_ok": int((merged_df["status"].astype(str) == "ok").sum()) if "status" in merged_df.columns else int(len(merged_df)),
        "n_with_survival_any": int(merged_df["survival_matched_any"].sum()),
        "n_with_survival_os": int(merged_df["survival_matched_os"].sum()),
        "n_with_survival_dss": int(merged_df["survival_matched_dss"].sum()),
        "n_with_survival_dfs": int(merged_df["survival_matched_dfs"].sum()),
        "n_with_clinical": int(merged_df["clinical_matched"].sum()),
        "n_with_radiomics": int(merged_df["radiomics_matched"].sum()),
    }
    summary_path = os.path.join(args.out_dir, "preparation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] wrote stage-2 metafile: {out_csv_path}")
    print(
        f"[summary] patients={summary['n_patients']} "
        f"survival_any={summary['n_with_survival_any']} "
        f"os={summary['n_with_survival_os']} "
        f"dss={summary['n_with_survival_dss']} "
        f"dfs={summary['n_with_survival_dfs']} "
        f"clinical={summary['n_with_clinical']} "
        f"radiomics={summary['n_with_radiomics']}"
    )
    print(f"[summary] wrote {summary_path}")


if __name__ == "__main__":
    main()
