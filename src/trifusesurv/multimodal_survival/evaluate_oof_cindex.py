#!/usr/bin/env python3
"""Compute OOF c-index from exported per-fold risk CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from trifusesurv.utils.clinical import ENDPOINT_MAP
from trifusesurv.utils.survival import concordance_index


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta_csv", required=True)
    p.add_argument("--id_col", type=str, default="patient_id")
    p.add_argument("--endpoint", type=str, default="OS", choices=["OS", "DSS", "DFS"])
    p.add_argument("--time_col", type=str, default="")
    p.add_argument("--event_col", type=str, default="")
    p.add_argument("--risk_csv", action="append", default=[])
    p.add_argument("--trial_root", type=str, default="")
    p.add_argument("--exp_prefix", type=str, default="")
    p.add_argument("--weights", type=str, default="ema", choices=["last", "ema", "swa", "best"])
    p.add_argument("--risk_col", type=str, default="risk_score")
    p.add_argument("--out_json", type=str, default="")
    p.add_argument("--out_csv", type=str, default="")
    return p.parse_args()


def resolve_risk_files(args) -> List[Path]:
    files: List[Path] = []
    if args.risk_csv:
        files.extend(Path(p) for p in args.risk_csv)
    elif args.trial_root:
        root = Path(args.trial_root)
        pattern = f"{args.exp_prefix}_fold*/fold_*/test_risks_{args.weights}.csv" if args.exp_prefix else f"*/fold_*/test_risks_{args.weights}.csv"
        files.extend(sorted(root.glob(pattern)))
    files = [p for p in files if p.is_file()]
    if not files:
        raise FileNotFoundError("No risk CSV files found. Provide --risk_csv or --trial_root/--exp_prefix.")
    return files


def load_risks(files: List[Path], id_col: str, risk_col: str) -> pd.DataFrame:
    dfs = []
    for p in files:
        df = pd.read_csv(p)
        if id_col not in df.columns or risk_col not in df.columns:
            raise ValueError(f"Risk CSV missing required columns: {p}")
        sub = df[[id_col, risk_col]].copy()
        if "risk_endpoint" in df.columns:
            sub["risk_endpoint"] = df["risk_endpoint"].astype(str).str.upper()
        if "risk_horizon_days" in df.columns:
            sub["risk_horizon_days"] = pd.to_numeric(df["risk_horizon_days"], errors="coerce")
        sub[id_col] = sub[id_col].astype(str)
        sub["source_risk_csv"] = str(p)
        dfs.append(sub)
    out = pd.concat(dfs, axis=0, ignore_index=True)
    dup = out[id_col].duplicated(keep=False)
    if dup.any():
        dup_ids = out.loc[dup, id_col].drop_duplicates().tolist()
        raise ValueError(f"Duplicate patient IDs across risk CSVs: {dup_ids[:20]}")
    return out


def main():
    args = parse_args()

    if args.time_col == "" or args.event_col == "":
        tcol, ecol = ENDPOINT_MAP[args.endpoint]
        if args.time_col == "":
            args.time_col = tcol
        if args.event_col == "":
            args.event_col = ecol

    risk_files = resolve_risk_files(args)
    risk_df = load_risks(risk_files, args.id_col, args.risk_col)
    if "risk_endpoint" in risk_df.columns:
        if risk_df["risk_endpoint"].isna().any():
            raise ValueError("Risk CSVs are mixed: some include risk_endpoint and some do not.")
        endpoints = sorted(set(risk_df["risk_endpoint"].dropna().tolist()))
        if endpoints and (endpoints != [str(args.endpoint).upper()]):
            raise ValueError(f"Risk CSV endpoint mismatch: expected {args.endpoint}, found {endpoints}")
    if "risk_horizon_days" in risk_df.columns:
        if risk_df["risk_horizon_days"].isna().any():
            raise ValueError("Risk CSVs are mixed: some include risk_horizon_days and some do not.")
        horizons = sorted(set(float(x) for x in risk_df["risk_horizon_days"].tolist()))
        if len(horizons) > 1:
            raise ValueError(f"Risk CSV horizon mismatch: found multiple risk_horizon_days values {horizons}")

    meta = pd.read_csv(args.meta_csv, dtype={args.id_col: str})
    meta[args.id_col] = meta[args.id_col].astype(str)
    meta[args.time_col] = pd.to_numeric(meta[args.time_col], errors="coerce")
    meta[args.event_col] = pd.to_numeric(meta[args.event_col], errors="coerce")

    merged = meta[[args.id_col, args.time_col, args.event_col]].merge(risk_df, on=args.id_col, how="inner")
    merged = merged.dropna(subset=[args.time_col, args.event_col, args.risk_col]).copy()
    merged[args.event_col] = merged[args.event_col].astype(int)

    c_index = concordance_index(
        merged[args.time_col].to_numpy(dtype=float),
        merged[args.event_col].to_numpy(dtype=float),
        merged[args.risk_col].to_numpy(dtype=float),
    )

    summary = {
        "meta_csv": str(args.meta_csv),
        "endpoint": args.endpoint,
        "time_col": args.time_col,
        "event_col": args.event_col,
        "weights": args.weights,
        "n_risk_files": int(len(risk_files)),
        "n_predictions": int(len(risk_df)),
        "n_evaluable": int(len(merged)),
        "c_index": float(c_index),
        "risk_files": [str(p) for p in risk_files],
    }

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_csv, index=False)

    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
