#!/usr/bin/env python3
"""
trifusesurv.preprocessing.export_swinunetr

End-to-end OFFLINE preprocessing for Swin UNETR survival on OPSCC:

Inputs
------
1) DICOM cohort root (one folder per patient):
    OPSCC/A0522/*.dcm  (CT + RTSTRUCT in same folder)
2) Survival table:
    opscc_survival_time_event.csv
3) (Optional) A cohort_meta.csv produced by the summarization step is NOT required.
   This script will redo the minimal DICOM/RTSTRUCT parsing itself.

Outputs
-------
out_root/
  A0522/
    ct.nii.gz                 (float32, scaled to [0,1])
    mask_union.nii.gz         (uint8, tumor burden = primary ∪ nodal)
    mask_primary.nii.gz       (uint8; may be empty)
    mask_nodal.nii.gz         (uint8; may be empty)
    meta.json                 (per-case metadata + QC)
  ...
  cohort_preprocessed.csv     (training metafile for Swin UNETR survival)

Key design choices (Swin UNETR oriented)
----------------------------------------
- Fixed orientation: RAS
- Fixed spacing: default 1.5 x 1.5 x 2.0 mm (change via --spacing)
- Fixed model input size: default 96 x 96 x 96 (change via --size)
- Crop policy: union-mask bbox + margin_mm (default 20mm). If union mask cannot be built,
  fall back to foreground crop on CT intensity (simple, robust).
- ROI handling: NO primary-only assumption.
  ROI names are grouped into primary/nodal/excluded via patterns including "PT" and "MetaLymphnode".

Robustness vs your failure case (A5856-like)
--------------------------------------------
Sometimes rt-utils returns ROI mask arrays whose shape cannot be aligned to CT.
This script attempts:
  1) strict alignment via permutations (fast path)
  2) if fails, tries raw-mask path ONLY for bbox/voxel counts (not exported as NIfTI mask)
     and falls back to CT foreground crop (so the pipeline still produces ct.nii.gz).
In other words: it never crashes the whole cohort; it writes status/error/QC.

Dependencies
------------
pip install -U pydicom SimpleITK numpy pandas rt-utils
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
pip install opencv-python-headless

Example run
-----------
PYTHONPATH=TriFuseSurv_package/src python3 -m trifusesurv.preprocessing.export_swinunetr \
  --root OPSCC \
  --surv_csv opscc_survival_time_event.csv \
  --out_root OPSCC_preprocessed_128_2 \
  --spacing 0.5 0.5 1 \
  --size 128 128 128 \
  --margin_mm 30 \
  --hu_min -1000 --hu_max 1000
"""

import os
import json
import argparse
import itertools
import re
from typing import Optional, Tuple, List, Dict, Any

try:
    import numpy as np
    import pandas as pd
    import pydicom
    import SimpleITK as sitk
    from rt_utils import RTStructBuilder
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "a required dependency")
    raise SystemExit(
        f"Missing preprocessing dependency: {missing}. "
        "Run ./scripts/install_env.sh or install the package dependencies with "
        "`python -m pip install --upgrade -e .`."
    ) from exc
except ImportError as exc:
    msg = str(exc)
    if "libxcb" in msg or "cv2" in msg:
        raise SystemExit(
            "OpenCV failed to import for preprocessing. "
            "Reinstall the headless OpenCV build with:\n"
            "  python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless\n"
            "  python -m pip install --upgrade --no-cache-dir --force-reinstall opencv-python-headless==4.10.0.84"
        ) from exc
    raise


# ---------------------------
# Utilities: IDs + survival
# ---------------------------
SURVIVAL_COLUMNS = [
    "Patient_ID",
    "OS.TIME",
    "OS.EVENT",
    "DSS.TIME",
    "DSS.EVENT",
    "DFS.TIME",
    "DFS.EVENT",
]
SURVIVAL_VALUE_COLUMNS = SURVIVAL_COLUMNS[1:]


def norm_patient_id(pid: str) -> str:
    if pid is None:
        return ""
    s = str(pid).strip()
    m = re.match(r"^([A-Za-z]+)0*([0-9]+)$", s)
    if m:
        return f"{m.group(1).upper()}{int(m.group(2))}"
    return s.upper()


def load_survival_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = SURVIVAL_COLUMNS
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Survival CSV missing columns: {miss}")
    df = df.copy()
    if df["Patient_ID"].isna().any():
        raise ValueError("Survival CSV contains missing Patient_ID values.")
    df["patient_id_norm"] = df["Patient_ID"].astype(str).map(norm_patient_id)
    if (df["patient_id_norm"] == "").any():
        bad_rows = df.index[df["patient_id_norm"] == ""].tolist()[:10]
        raise ValueError(
            "Survival CSV contains blank Patient_ID values after normalization. "
            f"Example row indices: {bad_rows}"
        )
    for c in needed[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_surv_map(df_surv: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    if df_surv.empty:
        return {}

    dup_mask = df_surv["patient_id_norm"].duplicated(keep=False)
    if dup_mask.any():
        dup_df = df_surv.loc[dup_mask, ["Patient_ID", "patient_id_norm", *SURVIVAL_VALUE_COLUMNS]].copy()
        conflicting = []
        for pid_norm, grp in dup_df.groupby("patient_id_norm", sort=False):
            payloads = grp[SURVIVAL_VALUE_COLUMNS].drop_duplicates()
            if len(payloads) > 1:
                source_ids = sorted(grp["Patient_ID"].astype(str).unique().tolist())
                conflicting.append(f"{pid_norm}<-{source_ids}")
        if conflicting:
            preview = "; ".join(conflicting[:5])
            raise ValueError(
                "Survival CSV contains duplicate normalized patient IDs with conflicting labels: "
                f"{preview}"
            )

        before = len(df_surv)
        df_surv = df_surv.drop_duplicates(subset=["patient_id_norm"], keep="first").copy()
        print(f"[warn] collapsed {before - len(df_surv)} duplicate survival row(s) after ID normalization.")

    mp: Dict[str, Dict[str, Any]] = {}
    for _, r in df_surv.iterrows():
        k = str(r["patient_id_norm"])
        mp[k] = {col: r.get(col, np.nan) for col in SURVIVAL_VALUE_COLUMNS}
    return mp


# ---------------------------
# DICOM / RTSTRUCT discovery + CT loading
# ---------------------------
def list_patient_dirs(root: str, recursive: bool) -> List[str]:
    root = os.path.abspath(root)
    lvl1 = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if not recursive:
        return lvl1
    expanded = []
    for d in lvl1:
        subs = [os.path.join(d, s) for s in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, s))]
        expanded.extend(subs if subs else [d])
    # unique preserve order
    seen = set()
    out = []
    for p in expanded:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def safe_get_patient_id(patient_dir: str) -> str:
    return os.path.basename(os.path.normpath(patient_dir))


def find_rtstruct_path(dicom_dir: str) -> str:
    candidates = []
    for fn in os.listdir(dicom_dir):
        if not fn.lower().endswith(".dcm"):
            continue
        path = os.path.join(dicom_dir, fn)
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            if getattr(ds, "Modality", "") == "RTSTRUCT":
                candidates.append(path)
        except Exception:
            pass
    if not candidates:
        for fn in os.listdir(dicom_dir):
            if fn.upper().startswith("RS") and fn.lower().endswith(".dcm"):
                candidates.append(os.path.join(dicom_dir, fn))
    if not candidates:
        raise FileNotFoundError(f"No RTSTRUCT (RS*.dcm) found in {dicom_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def get_referenced_series_uid(rtstruct_path: str) -> Optional[str]:
    try:
        ds = pydicom.dcmread(rtstruct_path, stop_before_pixels=True)
        ref = ds.ReferencedFrameOfReferenceSequence[0] \
               .RTReferencedStudySequence[0] \
               .RTReferencedSeriesSequence[0]
        return ref.SeriesInstanceUID
    except Exception:
        return None


def load_ct_image(dicom_dir: str, prefer_series_uid: Optional[str] = None) -> Tuple[sitk.Image, str, int]:
    """
    Returns (ct_img, chosen_series_uid, num_slices).
    """
    reader = sitk.ImageSeriesReader()
    series_ids = list(reader.GetGDCMSeriesIDs(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found under {dicom_dir}")

    ct_series: List[Tuple[str, int]] = []
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(dicom_dir, sid)
        if not files:
            continue
        try:
            ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
            if getattr(ds0, "Modality", "") != "CT":
                continue
        except Exception:
            continue
        ct_series.append((sid, len(files)))

    if not ct_series:
        raise RuntimeError("Found series, but none with Modality=CT")

    if prefer_series_uid and any(sid == prefer_series_uid for sid, _ in ct_series):
        chosen_uid = prefer_series_uid
    else:
        ct_series.sort(key=lambda t: t[1], reverse=True)
        chosen_uid = ct_series[0][0]

    files = reader.GetGDCMSeriesFileNames(dicom_dir, chosen_uid)
    reader.SetFileNames(files)
    # reduce ITK warning noise; also can help on weird series
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    img = reader.Execute()
    return img, chosen_uid, len(files)


# ---------------------------
# ROI grouping (primary / nodal / exclude)
# ---------------------------
def _norm(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in s).strip()


def is_excluded_roi(name: str) -> bool:
    n = _norm(name)
    bad = [
        "ctv", "ptv", "itv",
        "body", "external", "skin",
        "ring", "dose", "bolus", "avoid", "couch",
        "brainstem", "spinal", "cord", "parotid", "submandibular", "larynx",
        "mandible", "optic", "lens", "chiasm",
    ]
    return any(b in n for b in bad)


def roi_group(name: str) -> str:
    n = _norm(name)
    if is_excluded_roi(name):
        return "exclude"

    # tuned for your cohort: PT + MetaLymphnode
    nodal_keys = ["gtvn", "node", "nodal", "ln", "lymph", "level", "neck", "metalymphnode"]
    if any(k in n for k in nodal_keys):
        return "nodal"

    primary_keys = ["gtvp", "primary", "prim", "pt", "tumor", "lesion", "mass", "target"]
    if any(k in n for k in primary_keys):
        return "primary"

    return "other"


# ---------------------------
# Mask extraction + alignment
# ---------------------------
def mask_sitk_from_rt_strict(rtstruct: RTStructBuilder, roi_name: str, ct_img: sitk.Image) -> sitk.Image:
    """
    Strict: requires the ROI mask array can be permuted to (z,y,x)=CT.
    """
    arr = rtstruct.get_roi_mask_by_name(roi_name).astype(np.uint8)
    ct_x, ct_y, ct_z = ct_img.GetSize()
    target = (ct_z, ct_y, ct_x)

    if arr.shape == target:
        arr_zyx = arr
    else:
        arr_zyx = None
        for perm in itertools.permutations((0, 1, 2), 3):
            cand = arr.transpose(perm)
            if cand.shape == target:
                arr_zyx = cand
                break
        if arr_zyx is None:
            raise RuntimeError(
                f"Could not align ROI '{roi_name}' mask to CT. "
                f"CT (x,y,z)={(ct_x,ct_y,ct_z)} expected (z,y,x)={target} raw={arr.shape}"
            )

    m = sitk.GetImageFromArray(arr_zyx)
    m = sitk.Cast(m, sitk.sitkUInt8)
    m.CopyInformation(ct_img)
    if m.GetSize() != ct_img.GetSize():
        raise RuntimeError(f"Mask/CT size mismatch after alignment: mask={m.GetSize()} ct={ct_img.GetSize()}")
    return m


def build_group_mask(
    rtstruct: RTStructBuilder,
    roi_names: List[str],
    ct_img: sitk.Image,
) -> Tuple[Optional[sitk.Image], Dict[str, str]]:
    """
    Returns (union_mask_sitk or None, per_roi_qc dict).

    We only build/export masks if all included ROIs can be strictly aligned.
    If any ROI in the group fails strict alignment, we return None for that group
    (and record qc). This prevents exporting geometrically wrong masks.
    """
    qc = {}
    masks = []
    for n in roi_names:
        try:
            m = mask_sitk_from_rt_strict(rtstruct, n, ct_img)
            masks.append(m)
            qc[n] = "aligned"
        except Exception as ex:
            qc[n] = f"failed({type(ex).__name__})"
            # do NOT partially build a union with mixed aligned/unaligned
            return None, qc

    if not masks:
        return None, qc

    out = sitk.Cast(masks[0] > 0, sitk.sitkUInt8)
    for m in masks[1:]:
        out = sitk.Or(out, sitk.Cast(m > 0, sitk.sitkUInt8))
    out.CopyInformation(ct_img)
    return out, qc


# ---------------------------
# Preprocessing primitives (SITK)
# ---------------------------
def orient_to_RAS(img: sitk.Image) -> sitk.Image:
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("RAS")
    return f.Execute(img)


def resample_to_spacing(img: sitk.Image, out_spacing=(1.5, 1.5, 2.0), is_label=False) -> sitk.Image:
    out_spacing = tuple(float(x) for x in out_spacing)
    in_spacing = img.GetSpacing()
    in_size = img.GetSize()
    out_size = [int(np.round(in_size[i] * (in_spacing[i] / out_spacing[i]))) for i in range(3)]

    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing(out_spacing)
    res.SetSize(out_size)
    res.SetOutputDirection(img.GetDirection())
    res.SetOutputOrigin(img.GetOrigin())
    res.SetTransform(sitk.Transform())
    res.SetDefaultPixelValue(0)

    if is_label:
        res.SetInterpolator(sitk.sitkNearestNeighbor)
        img = sitk.Cast(img, sitk.sitkUInt8)
    else:
        res.SetInterpolator(sitk.sitkLinear)
        img = sitk.Cast(img, sitk.sitkFloat32)

    return res.Execute(img)


def clip_and_scale_hu(ct: sitk.Image, hu_min=-1000.0, hu_max=1000.0) -> sitk.Image:
    ct = sitk.Cast(ct, sitk.sitkFloat32)
    ct = sitk.Clamp(ct, lowerBound=float(hu_min), upperBound=float(hu_max))
    ct = (ct - float(hu_min)) / (float(hu_max) - float(hu_min))
    return ct


def bbox_from_mask(mask: sitk.Image) -> Tuple[int, int, int, int, int, int]:
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    if not stats.HasLabel(1):
        raise RuntimeError("Mask has no positive voxels.")
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    return x, x + sx, y, y + sy, z, z + sz


def crop_with_margin(img: sitk.Image, bbox, margin_mm: float) -> sitk.Image:
    x0, x1, y0, y1, z0, z1 = bbox
    spx, spy, spz = img.GetSpacing()
    mx = int(np.round(margin_mm / spx))
    my = int(np.round(margin_mm / spy))
    mz = int(np.round(margin_mm / spz))
    X, Y, Z = img.GetSize()
    x0 = max(0, x0 - mx); x1 = min(X, x1 + mx)
    y0 = max(0, y0 - my); y1 = min(Y, y1 + my)
    z0 = max(0, z0 - mz); z1 = min(Z, z1 + mz)
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([x0, y0, z0])
    roi.SetSize([x1 - x0, y1 - y0, z1 - z0])
    return roi.Execute(img)


def pad_or_crop_to_size(img: sitk.Image, out_size=(96, 96, 96), is_label=False) -> sitk.Image:
    out_size = tuple(int(x) for x in out_size)
    in_size = img.GetSize()

    pad_lower = [0, 0, 0]
    pad_upper = [0, 0, 0]
    for i in range(3):
        if in_size[i] < out_size[i]:
            diff = out_size[i] - in_size[i]
            pad_lower[i] = diff // 2
            pad_upper[i] = diff - pad_lower[i]

    if any(p > 0 for p in pad_lower + pad_upper):
        pad = sitk.ConstantPadImageFilter()
        pad.SetPadLowerBound(pad_lower)
        pad.SetPadUpperBound(pad_upper)
        pad.SetConstant(0)
        img = pad.Execute(img)

    in2 = img.GetSize()
    start = [0, 0, 0]
    size = [out_size[0], out_size[1], out_size[2]]
    for i in range(3):
        if in2[i] > out_size[i]:
            start[i] = (in2[i] - out_size[i]) // 2

    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex(start)
    roi.SetSize(size)
    img = roi.Execute(img)

    return sitk.Cast(img, sitk.sitkUInt8 if is_label else sitk.sitkFloat32)


def crop_foreground_ct(ct01: sitk.Image, thr: float = 0.05) -> sitk.Image:
    """
    Fallback crop: threshold CT (already in [0,1]) and crop to largest connected foreground bbox.
    Simple, robust when RTSTRUCT mask can't be aligned.
    """
    m = sitk.Cast(ct01 > float(thr), sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(m)
    rel = sitk.RelabelComponent(cc, sortByObjectSize=True)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(rel)
    if not stats.HasLabel(1):
        return ct01  # no crop
    x, y, z, sx, sy, sz = stats.GetBoundingBox(1)
    roi = sitk.RegionOfInterestImageFilter()
    roi.SetIndex([x, y, z])
    roi.SetSize([sx, sy, sz])
    return roi.Execute(ct01)


# ---------------------------
# Per-patient processing
# ---------------------------
def process_patient(
    patient_dir: str,
    out_root: str,
    surv_map: Optional[Dict[str, Dict[str, Any]]],
    spacing: Tuple[float, float, float],
    out_size: Tuple[int, int, int],
    margin_mm: float,
    hu_min: float,
    hu_max: float,
) -> Dict[str, Any]:
    pid = safe_get_patient_id(patient_dir)
    pid_norm = norm_patient_id(pid)
    out_dir = os.path.join(out_root, pid)
    os.makedirs(out_dir, exist_ok=True)

    meta: Dict[str, Any] = {
        "patient_id": pid,
        "patient_id_norm": pid_norm,
        "patient_dir": patient_dir,
        "out_dir": out_dir,
        "status": "ok",
        "error": "",

        "rtstruct_path": "",
        "rtstruct_referenced_series_uid": "",
        "ct_series_uid": "",
        "ct_num_slices": np.nan,
        "ct_size_xyz": "",
        "ct_spacing_xyz": "",

        "roi_names_all": "",
        "roi_primary_names": "",
        "roi_nodal_names": "",
        "roi_other_names": "",
        "roi_excluded_names": "",

        "mask_union_qc": "",
        "mask_primary_qc": "",
        "mask_nodal_qc": "",

        "ct_out_path": os.path.join(out_dir, "ct.nii.gz"),
        "mask_union_out_path": os.path.join(out_dir, "mask_union.nii.gz"),
        "mask_primary_out_path": os.path.join(out_dir, "mask_primary.nii.gz"),
        "mask_nodal_out_path": os.path.join(out_dir, "mask_nodal.nii.gz"),

        "target_spacing": list(spacing),
        "target_size": list(out_size),
        "margin_mm": float(margin_mm),
        "hu_min": float(hu_min),
        "hu_max": float(hu_max),

        # survival
        "OS.TIME": np.nan, "OS.EVENT": np.nan,
        "DSS.TIME": np.nan, "DSS.EVENT": np.nan,
        "DFS.TIME": np.nan, "DFS.EVENT": np.nan,
    }

    try:
        rt_path = find_rtstruct_path(patient_dir)
        meta["rtstruct_path"] = rt_path
        ref_uid = get_referenced_series_uid(rt_path)
        meta["rtstruct_referenced_series_uid"] = ref_uid or ""

        ct_img, ct_uid, n_slices = load_ct_image(patient_dir, prefer_series_uid=ref_uid)
        meta["ct_series_uid"] = ct_uid
        meta["ct_num_slices"] = int(n_slices)
        meta["ct_size_xyz"] = "|".join(map(str, ct_img.GetSize()))
        meta["ct_spacing_xyz"] = "|".join([f"{x:.6g}" for x in ct_img.GetSpacing()])

        # Load rtstruct + ROI names
        rtstruct = RTStructBuilder.create_from(dicom_series_path=patient_dir, rt_struct_path=rt_path)
        roi_names = list(rtstruct.get_roi_names())
        meta["roi_names_all"] = "|".join(roi_names)

        primary = [r for r in roi_names if roi_group(r) == "primary"]
        nodal = [r for r in roi_names if roi_group(r) == "nodal"]
        other = [r for r in roi_names if roi_group(r) == "other"]
        excl = [r for r in roi_names if roi_group(r) == "exclude"]

        meta["roi_primary_names"] = "|".join(primary)
        meta["roi_nodal_names"] = "|".join(nodal)
        meta["roi_other_names"] = "|".join(other)
        meta["roi_excluded_names"] = "|".join(excl)

        # Build strict-aligned masks (export only if aligned)
        mask_primary, qc_p = build_group_mask(rtstruct, primary, ct_img)
        mask_nodal, qc_n = build_group_mask(rtstruct, nodal, ct_img)

        # union only if at least one group mask exists
        mask_union = None
        if mask_primary is not None and mask_nodal is not None:
            mask_union = sitk.Or(mask_primary > 0, mask_nodal > 0)
            mask_union = sitk.Cast(mask_union, sitk.sitkUInt8)
            mask_union.CopyInformation(ct_img)
            meta["mask_union_qc"] = "primary+nodal(aligned)"
        elif mask_primary is not None:
            mask_union = sitk.Cast(mask_primary > 0, sitk.sitkUInt8)
            mask_union.CopyInformation(ct_img)
            meta["mask_union_qc"] = "primary_only(aligned)"
        elif mask_nodal is not None:
            mask_union = sitk.Cast(mask_nodal > 0, sitk.sitkUInt8)
            mask_union.CopyInformation(ct_img)
            meta["mask_union_qc"] = "nodal_only(aligned)"
        else:
            meta["mask_union_qc"] = "no_aligned_mask"

        meta["mask_primary_qc"] = ";".join([f"{k}:{v}" for k, v in qc_p.items()]) if qc_p else "empty"
        meta["mask_nodal_qc"] = ";".join([f"{k}:{v}" for k, v in qc_n.items()]) if qc_n else "empty"

        # ---- Preprocess CT + masks ----
        # 1) orient to RAS
        ct_img = orient_to_RAS(ct_img)
        if mask_primary is not None: mask_primary = orient_to_RAS(mask_primary)
        if mask_nodal is not None: mask_nodal = orient_to_RAS(mask_nodal)
        if mask_union is not None: mask_union = orient_to_RAS(mask_union)

        # 2) resample to target spacing
        ct_img = resample_to_spacing(ct_img, out_spacing=spacing, is_label=False)
        if mask_primary is not None: mask_primary = resample_to_spacing(mask_primary, out_spacing=spacing, is_label=True)
        if mask_nodal is not None: mask_nodal = resample_to_spacing(mask_nodal, out_spacing=spacing, is_label=True)
        if mask_union is not None: mask_union = resample_to_spacing(mask_union, out_spacing=spacing, is_label=True)

        # 3) HU clip + scale -> [0,1]
        ct01 = clip_and_scale_hu(ct_img, hu_min=hu_min, hu_max=hu_max)

        # 4) crop
        if mask_union is not None:
            bb = bbox_from_mask(mask_union)
            ct01 = crop_with_margin(ct01, bb, margin_mm=margin_mm)
            mask_union = crop_with_margin(mask_union, bb, margin_mm=margin_mm)
            if mask_primary is not None: mask_primary = crop_with_margin(mask_primary, bb, margin_mm=margin_mm)
            if mask_nodal is not None: mask_nodal = crop_with_margin(mask_nodal, bb, margin_mm=margin_mm)
            meta["crop_mode"] = "union_bbox"
        else:
            # fallback crop on CT foreground (already scaled)
            ct01 = crop_foreground_ct(ct01, thr=0.05)
            meta["crop_mode"] = "ct_foreground"

        # 5) pad/crop to fixed size
        ct01 = pad_or_crop_to_size(ct01, out_size=out_size, is_label=False)
        if mask_union is not None:
            mask_union = pad_or_crop_to_size(mask_union, out_size=out_size, is_label=True)
        if mask_primary is not None:
            mask_primary = pad_or_crop_to_size(mask_primary, out_size=out_size, is_label=True)
        if mask_nodal is not None:
            mask_nodal = pad_or_crop_to_size(mask_nodal, out_size=out_size, is_label=True)

        # 6) write outputs
        sitk.WriteImage(ct01, meta["ct_out_path"], useCompression=True)

        # If no aligned mask, write empty masks (so training pipeline stays uniform)
        def empty_mask_like(img: sitk.Image) -> sitk.Image:
            zyx = sitk.GetArrayFromImage(img)
            e = np.zeros_like(zyx, dtype=np.uint8)
            m = sitk.GetImageFromArray(e)
            m.CopyInformation(img)
            return sitk.Cast(m, sitk.sitkUInt8)

        if mask_union is None:
            mask_union = empty_mask_like(ct01)
        if mask_primary is None:
            mask_primary = empty_mask_like(ct01)
        if mask_nodal is None:
            mask_nodal = empty_mask_like(ct01)

        sitk.WriteImage(mask_union, meta["mask_union_out_path"], useCompression=True)
        sitk.WriteImage(mask_primary, meta["mask_primary_out_path"], useCompression=True)
        sitk.WriteImage(mask_nodal, meta["mask_nodal_out_path"], useCompression=True)

        # volume stats after preprocessing (in voxel counts; spacing fixed)
        vv = float(np.prod(spacing))
        mu = sitk.GetArrayFromImage(mask_union).astype(np.uint8)
        mp = sitk.GetArrayFromImage(mask_primary).astype(np.uint8)
        mn = sitk.GetArrayFromImage(mask_nodal).astype(np.uint8)

        meta["mask_union_voxels"] = int(mu.sum())
        meta["mask_primary_voxels"] = int(mp.sum())
        meta["mask_nodal_voxels"] = int(mn.sum())

        meta["mask_union_mm3"] = float(meta["mask_union_voxels"]) * vv
        meta["mask_primary_mm3"] = float(meta["mask_primary_voxels"]) * vv
        meta["mask_nodal_mm3"] = float(meta["mask_nodal_voxels"]) * vv

        # attach survival
        if surv_map is not None and pid_norm in surv_map:
            s = surv_map[pid_norm]
            for k in ["OS.TIME", "OS.EVENT", "DSS.TIME", "DSS.EVENT", "DFS.TIME", "DFS.EVENT"]:
                meta[k] = s.get(k, np.nan)
            meta["survival_matched"] = True
        else:
            meta["survival_matched"] = False

        # write per-case meta
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    except Exception as ex:
        meta["status"] = "error"
        meta["error"] = f"{type(ex).__name__}: {ex}"
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return meta


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DICOM cohort root (one folder per patient)")
    ap.add_argument("--surv_csv", required=True, help="opscc_survival_time_event.csv path")
    ap.add_argument("--out_root", required=True, help="output root for preprocessed NIfTI")
    ap.add_argument("--out_csv", default="cohort_preprocessed.csv", help="output CSV name (under out_root)")
    ap.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 2.0])
    ap.add_argument("--size", type=int, nargs=3, default=[96, 96, 96])
    ap.add_argument("--margin_mm", type=float, default=20.0)
    ap.add_argument("--hu_min", type=float, default=-1000.0)
    ap.add_argument("--hu_max", type=float, default=1000.0)
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    df_surv = load_survival_table(args.surv_csv)
    surv_map = build_surv_map(df_surv)
    print(f"[info] loaded survival rows={len(df_surv)} unique_ids={len(surv_map)}")

    patient_dirs = list_patient_dirs(root, recursive=args.recursive)
    print(f"[info] found {len(patient_dirs)} patient folders under {root}")

    metas: List[Dict[str, Any]] = []
    for i, pdir in enumerate(patient_dirs, 1):
        pid = safe_get_patient_id(pdir)
        print(f"[{i:05d}/{len(patient_dirs):05d}] {pid}")
        meta = process_patient(
            patient_dir=pdir,
            out_root=out_root,
            surv_map=surv_map,
            spacing=tuple(args.spacing),
            out_size=tuple(args.size),
            margin_mm=float(args.margin_mm),
            hu_min=float(args.hu_min),
            hu_max=float(args.hu_max),
        )
        metas.append(meta)

    df = pd.DataFrame(metas)

    # order columns for training
    preferred = [
        "patient_id", "patient_id_norm", "status", "error", "survival_matched",
        "OS.TIME", "OS.EVENT", "DSS.TIME", "DSS.EVENT", "DFS.TIME", "DFS.EVENT",
        "ct_out_path", "mask_union_out_path", "mask_primary_out_path", "mask_nodal_out_path",
        "mask_union_voxels", "mask_primary_voxels", "mask_nodal_voxels",
        "mask_union_mm3", "mask_primary_mm3", "mask_nodal_mm3",
        "crop_mode", "mask_union_qc", "mask_primary_qc", "mask_nodal_qc",
        "roi_names_all", "roi_primary_names", "roi_nodal_names", "roi_other_names", "roi_excluded_names",
        "ct_series_uid", "ct_num_slices", "ct_size_xyz", "ct_spacing_xyz",
        "target_spacing", "target_size", "margin_mm", "hu_min", "hu_max",
        "patient_dir", "rtstruct_path", "rtstruct_referenced_series_uid",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    out_csv = os.path.join(out_root, args.out_csv)
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote CSV: {out_csv}")

    n_ok = int((df["status"] == "ok").sum()) if "status" in df.columns else 0
    n_err = len(df) - n_ok
    n_surv = int(df["survival_matched"].sum()) if "survival_matched" in df.columns else 0
    print(f"[summary] ok={n_ok} error={n_err} survival_matched={n_surv}/{len(df)}")


if __name__ == "__main__":
    main()
