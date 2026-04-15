#!/usr/bin/env python3
"""
trifusesurv.preprocessing.make_cv_splits

Reproduces the packaged train/val/test splits used by the survival trainer:
- same status filtering
- same QC filtering
- same stratified K-fold construction
- same per-fold train/val split seeding

Outputs:
  out_dir/
    splits.csv                      (patient_id, fold, split)
    fold_00/train_ids.txt, val_ids.txt, test_ids.txt
    ...

PYTHONPATH=src python3 -m trifusesurv.preprocessing.make_cv_splits \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --qc_report OPSCC_preprocessed_128/qc/qc_report.csv \
  --qc_policy none --qc_drop_air_gt 0 \
  --endpoint OS \
  --cv_folds 4 --val_frac 0.2 --split_seed 1 \
  --out_dir runs/opscc_splits_<endpoint>_seed1

PYTHONPATH=src python3 -m trifusesurv.preprocessing.make_cv_splits \
  --meta_csv OPSCC_preprocessed_128/cohort_preprocessed.csv \
  --qc_report OPSCC_preprocessed_128/qc/qc_report.csv \
  --qc_policy none --qc_drop_air_gt 0 \
  --endpoint OS \
  --cv_folds 1 --val_frac 0.2 --split_seed 1 \
  --out_dir runs/opscc_splits_<endpoint>_seed3

When `--cv_folds 1`, this writes a single `fold_00/` with train/val IDs and an empty
`test_ids.txt`, which keeps the downstream packaged trainer interface consistent.
"""

import os
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd


ENDPOINT_MAP = {"OS": ("OS.TIME", "OS.EVENT"), "DSS": ("DSS.TIME", "DSS.EVENT"), "DFS": ("DFS.TIME", "DFS.EVENT")}


def _validate_patient_ids(patient_ids: pd.Series, *, context: str) -> pd.Series:
    patient_ids = patient_ids.astype(str).str.strip()
    blank = patient_ids == ""
    if blank.any():
        raise ValueError(f"{context} contains blank patient_id values.")

    dup_ids = patient_ids[patient_ids.duplicated(keep=False)].unique().tolist()
    if dup_ids:
        raise ValueError(f"{context} contains duplicate patient_id values: {dup_ids[:10]}")
    return patient_ids


def _validate_binary_events(events, *, context: str) -> np.ndarray:
    arr = np.asarray(events, dtype=int)
    bad = sorted(int(x) for x in np.unique(arr) if int(x) not in (0, 1))
    if bad:
        raise ValueError(f"{context} must be binary with values in {{0,1}}, got {bad}")
    return arr


def _endpoint_valid_mask(df: pd.DataFrame, time_col: str, event_col: str) -> pd.Series:
    times = pd.to_numeric(df[time_col], errors="coerce")
    events = pd.to_numeric(df[event_col], errors="coerce")
    return times.notna() & events.notna() & (times > 0) & events.isin([0, 1])


def load_items_for_splits(
    meta_csv: str,
    endpoint: str,
    require_status_ok: bool = True,
    require_survival_matched: bool = True,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(meta_csv)

    if endpoint not in ENDPOINT_MAP:
        raise ValueError(f"--endpoint must be one of {list(ENDPOINT_MAP.keys())}, got {endpoint}")

    tcol, ecol = ENDPOINT_MAP[endpoint]
    needed = ["patient_id", tcol, ecol]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metafile missing required columns: {missing}")

    df = df.copy()

    if require_status_ok and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"]

    df[tcol] = pd.to_numeric(df[tcol], errors="coerce")
    df[ecol] = pd.to_numeric(df[ecol], errors="coerce")
    if require_survival_matched:
        df = df[df[tcol].notna() & df[ecol].notna()]
    df = df.dropna(subset=[tcol, ecol])
    df["patient_id"] = _validate_patient_ids(df["patient_id"], context=f"{meta_csv} after filtering")
    df[ecol] = df[ecol].astype(int)
    _validate_binary_events(df[ecol].to_numpy(), context=f"{meta_csv}:{ecol}")

    items: List[Dict[str, Any]] = []
    for _, r in df.iterrows():  # preserves file row order (matches your training script)
        items.append(
            {
                "patient_id": str(r["patient_id"]),
                "time": float(r[tcol]),
                "event": int(r[ecol]),
            }
        )
    return items


def load_primary_and_aux_train_ids(
    meta_csv: str,
    endpoint: str,
    require_status_ok: bool = True,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    df = pd.read_csv(meta_csv)

    if endpoint not in ENDPOINT_MAP:
        raise ValueError(f"--endpoint must be one of {list(ENDPOINT_MAP.keys())}, got {endpoint}")

    tcol, ecol = ENDPOINT_MAP[endpoint]
    needed = ["patient_id", *[c for pair in ENDPOINT_MAP.values() for c in pair]]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Metafile missing required columns: {missing}")

    df = df.copy()
    if require_status_ok and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower() == "ok"]

    df["patient_id"] = _validate_patient_ids(df["patient_id"], context=f"{meta_csv} after filtering")

    valid_any = pd.Series(False, index=df.index)
    for time_col, event_col in ENDPOINT_MAP.values():
        valid_ep = _endpoint_valid_mask(df, time_col, event_col)
        if bool(valid_ep.any()):
            _validate_binary_events(
                pd.to_numeric(df.loc[valid_ep, event_col], errors="raise").astype(int).to_numpy(),
                context=f"{meta_csv}:{event_col}",
            )
        valid_any = valid_any | valid_ep

    primary_valid = _endpoint_valid_mask(df, tcol, ecol)
    primary_items = []
    for _, r in df.loc[primary_valid].iterrows():
        primary_items.append(
            {
                "patient_id": str(r["patient_id"]),
                "time": float(r[tcol]),
                "event": int(pd.to_numeric(r[ecol], errors="raise")),
            }
        )

    aux_only_ids = (
        df.loc[valid_any & (~primary_valid), "patient_id"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return primary_items, aux_only_ids


def qc_filter_items(
    items: List[Dict[str, Any]],
    qc_report: str,
    qc_policy: str,
    qc_id_col: str,
    qc_severity_col: str,
    qc_drop_if_contains: List[str],
    qc_drop_air_gt: float,
) -> List[Dict[str, Any]]:
    if qc_policy == "none" or not qc_report:
        return items

    keep_ids = load_qc_keep_ids(
        qc_report=qc_report,
        qc_policy=qc_policy,
        qc_id_col=qc_id_col,
        qc_severity_col=qc_severity_col,
        qc_drop_if_contains=qc_drop_if_contains,
        qc_drop_air_gt=qc_drop_air_gt,
    )

    item_ids = set(str(it["patient_id"]) for it in items)
    qc_ids_all = load_qc_all_ids(qc_report, qc_id_col)
    missing_in_qc = item_ids - qc_ids_all
    if missing_in_qc:
        print(f"[qc] warning: {len(missing_in_qc)} item(s) not found in QC report; they will be dropped.")

    before = len(items)
    items2 = [it for it in items if str(it["patient_id"]) in keep_ids]
    print(f"[qc] applied {qc_policy}: kept {len(items2)}/{before} (dropped {before-len(items2)})")
    return items2


def load_qc_all_ids(qc_report: str, qc_id_col: str) -> set[str]:
    qc = pd.read_csv(qc_report)
    if qc_id_col not in qc.columns:
        raise ValueError(f"QC report missing '{qc_id_col}'")
    return set(qc[qc_id_col].astype(str).tolist())


def load_qc_keep_ids(
    *,
    qc_report: str,
    qc_policy: str,
    qc_id_col: str,
    qc_severity_col: str,
    qc_drop_if_contains: List[str],
    qc_drop_air_gt: float,
) -> set[str]:
    if qc_policy == "none" or not qc_report:
        return set()

    qc = pd.read_csv(qc_report)
    if qc_id_col not in qc.columns:
        raise ValueError(f"QC report missing '{qc_id_col}'")

    qc = qc.copy()
    qc[qc_id_col] = qc[qc_id_col].astype(str)

    if qc_severity_col not in qc.columns:
        raise ValueError(f"QC report must include '{qc_severity_col}' column.")
    sev = qc[qc_severity_col].astype(str).str.lower()

    if qc_policy == "drop_fail":
        keep = sev != "fail"
    elif qc_policy == "drop_fail_warn":
        keep = sev == "pass"
    else:
        raise ValueError(f"Unknown qc_policy: {qc_policy}")

    flag_col = None
    for c in ["all_flags", "flags", "fail_flags", "warn_flags", "outlier_flags"]:
        if c in qc.columns:
            flag_col = c
            break
    if qc_drop_if_contains and flag_col is not None:
        txt = qc[flag_col].fillna("").astype(str)
        for sub in qc_drop_if_contains:
            keep = keep & (~txt.str.contains(str(sub), regex=False))

    if qc_drop_air_gt > 0 and "union_in_air_frac" in qc.columns:
        air = pd.to_numeric(qc["union_in_air_frac"], errors="coerce")
        keep = keep & ~(air > float(qc_drop_air_gt))

    return set(qc.loc[keep, qc_id_col].astype(str).tolist())


def stratified_kfold_indices(events: np.ndarray, k: int, seed: int) -> List[List[int]]:
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rng = np.random.default_rng(seed)
    events = _validate_binary_events(events, context="events")

    idx_e = np.where(events == 1)[0]
    idx_c = np.where(events == 0)[0]
    rng.shuffle(idx_e)
    rng.shuffle(idx_c)

    folds = [[] for _ in range(k)]
    for i, idx in enumerate(idx_e):
        folds[i % k].append(int(idx))
    for i, idx in enumerate(idx_c):
        folds[i % k].append(int(idx))
    for f in folds:
        rng.shuffle(f)
    return folds


def stratified_train_val_split(indices: List[int], events: np.ndarray, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    if not 0 <= float(val_frac) < 1:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")

    rng = np.random.default_rng(seed)
    idx = np.array(indices, dtype=int)
    if len(idx) <= 1 or val_frac <= 0:
        return [int(x) for x in idx.tolist()], []
    ev = _validate_binary_events(events, context="events")[idx]

    idx_e = idx[ev == 1]
    idx_c = idx[ev == 0]
    rng.shuffle(idx_e)
    rng.shuffle(idx_c)

    n_val = int(round(len(idx) * val_frac))
    n_val = min(n_val, len(idx) - 1)
    if n_val <= 0:
        return [int(x) for x in idx.tolist()], []
    n_val_e = int(round(n_val * (len(idx_e) / max(1, len(idx)))))
    n_val_e = min(n_val_e, len(idx_e))
    n_val_c = min(n_val - n_val_e, len(idx_c))

    val_idx = np.concatenate([idx_e[:n_val_e], idx_c[:n_val_c]])
    rng.shuffle(val_idx)

    val_set = set(int(x) for x in val_idx.tolist())
    tr_idx = [int(x) for x in idx.tolist() if int(x) not in val_set]
    va_idx = [int(x) for x in idx.tolist() if int(x) in val_set]
    return tr_idx, va_idx


def make_fold_splits(events: np.ndarray, cv_folds: int, val_frac: float, split_seed: int) -> List[Dict[str, List[int]]]:
    events = _validate_binary_events(events, context="events")
    cv_folds = int(cv_folds)
    if cv_folds < 1:
        raise ValueError(f"cv_folds must be >= 1, got {cv_folds}")

    if cv_folds == 1:
        tr_idx, va_idx = stratified_train_val_split(list(range(len(events))), events, val_frac, split_seed + 1000)
        return [{"train": tr_idx, "val": va_idx, "test": []}]

    if len(events) < cv_folds:
        raise ValueError(f"Not enough samples ({len(events)}) for {cv_folds}-fold CV after QC.")

    folds = stratified_kfold_indices(events, cv_folds, split_seed)
    split_defs: List[Dict[str, List[int]]] = []
    for fold_idx in range(cv_folds):
        test_idx = folds[fold_idx]
        trainval_idx = [i for other_idx, fold in enumerate(folds) if other_idx != fold_idx for i in fold]
        tr_idx, va_idx = stratified_train_val_split(trainval_idx, events, val_frac, split_seed + 1000 + fold_idx)
        split_defs.append({"train": tr_idx, "val": va_idx, "test": test_idx})
    return split_defs


def write_ids(path: str, ids: List[str]):
    with open(path, "w") as f:
        for x in ids:
            f.write(f"{x}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", type=str, required=True)
    ap.add_argument("--endpoint", type=str, default="OS", choices=["OS", "DSS", "DFS"])

    # Match your training script flags
    ap.add_argument("--keep_bad_status", action="store_true")
    ap.add_argument("--keep_unmatched_survival", action="store_true")
    ap.add_argument("--include_aux_only_train", dest="include_aux_only_train", action="store_true")
    ap.add_argument("--no_include_aux_only_train", dest="include_aux_only_train", action="store_false")
    ap.set_defaults(include_aux_only_train=True)

    # QC
    ap.add_argument("--qc_report", type=str, default="")
    ap.add_argument("--qc_policy", type=str, default="none", choices=["none", "drop_fail", "drop_fail_warn"])
    ap.add_argument("--qc_id_col", type=str, default="patient_id")
    ap.add_argument("--qc_severity_col", type=str, default="severity")
    ap.add_argument("--qc_drop_if_contains", type=str, action="append", default=[])
    ap.add_argument("--qc_drop_air_gt", type=float, default=-1.0)

    # CV
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--split_seed", type=int, default=1)

    ap.add_argument("--out_dir", type=str, default="cv_splits")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    items = load_items_for_splits(
        args.meta_csv,
        args.endpoint,
        require_status_ok=not args.keep_bad_status,
        require_survival_matched=not args.keep_unmatched_survival,
    )

    items = qc_filter_items(
        items,
        qc_report=args.qc_report,
        qc_policy=args.qc_policy,
        qc_id_col=args.qc_id_col,
        qc_severity_col=args.qc_severity_col,
        qc_drop_if_contains=args.qc_drop_if_contains,
        qc_drop_air_gt=args.qc_drop_air_gt,
    )

    if not items:
        raise ValueError("No samples remain after filtering and QC.")

    aux_only_train_ids: List[str] = []
    if args.include_aux_only_train:
        _, aux_only_train_ids = load_primary_and_aux_train_ids(
            args.meta_csv,
            args.endpoint,
            require_status_ok=not args.keep_bad_status,
        )
        if args.qc_policy != "none" and args.qc_report:
            keep_ids = load_qc_keep_ids(
                qc_report=args.qc_report,
                qc_policy=args.qc_policy,
                qc_id_col=args.qc_id_col,
                qc_severity_col=args.qc_severity_col,
                qc_drop_if_contains=args.qc_drop_if_contains,
                qc_drop_air_gt=args.qc_drop_air_gt,
            )
            aux_only_train_ids = [pid for pid in aux_only_train_ids if pid in keep_ids]
        if aux_only_train_ids:
            print(f"[mtl] adding {len(aux_only_train_ids)} aux-only training case(s) with non-primary survival labels")

    events = np.array([it["event"] for it in items], dtype=int)
    split_defs = make_fold_splits(events, args.cv_folds, args.val_frac, args.split_seed)

    rows = []
    for f, split_def in enumerate(split_defs):
        train_ids = [items[i]["patient_id"] for i in split_def["train"]]
        val_ids = [items[i]["patient_id"] for i in split_def["val"]]
        test_ids = [items[i]["patient_id"] for i in split_def["test"]]
        if aux_only_train_ids:
            train_ids = train_ids + [pid for pid in aux_only_train_ids if pid not in set(train_ids)]

        print(f"[fold {f:02d}] train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

        fold_dir = os.path.join(args.out_dir, f"fold_{f:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        write_ids(os.path.join(fold_dir, "train_ids.txt"), train_ids)
        write_ids(os.path.join(fold_dir, "val_ids.txt"), val_ids)
        write_ids(os.path.join(fold_dir, "test_ids.txt"), test_ids)

        for pid in train_ids:
            rows.append(
                {
                    "patient_id": pid,
                    "fold": f,
                    "split": "train",
                    "cohort": "aux_only_train" if pid in set(aux_only_train_ids) else "primary",
                }
            )
        for pid in val_ids:
            rows.append({"patient_id": pid, "fold": f, "split": "val", "cohort": "primary"})
        for pid in test_ids:
            rows.append({"patient_id": pid, "fold": f, "split": "test", "cohort": "primary"})

    out_csv = os.path.join(args.out_dir, "splits.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}")


if __name__ == "__main__":
    main()
