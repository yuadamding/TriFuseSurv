#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

DICOM_ROOT="${DICOM_ROOT:-OPSCC}"
SURV_CSV="${SURV_CSV:-opscc_survival_time_event.csv}"
OUT_ROOT="${OUT_ROOT:-OPSCC_preprocessed_128}"
OUT_CSV="${OUT_CSV:-cohort_preprocessed.csv}"
SPACING_X="${SPACING_X:-0.5}"
SPACING_Y="${SPACING_Y:-0.5}"
SPACING_Z="${SPACING_Z:-1}"
SIZE_D="${SIZE_D:-128}"
SIZE_H="${SIZE_H:-256}"
SIZE_W="${SIZE_W:-256}"
MARGIN_MM="${MARGIN_MM:-30}"
HU_MIN="${HU_MIN:--1000}"
HU_MAX="${HU_MAX:-1000}"
RECURSIVE="${RECURSIVE:-0}"

EXTRA_ARGS=()
if [[ "$RECURSIVE" == "1" ]]; then
  EXTRA_ARGS+=(--recursive)
fi

python3 -m trifusesurv.preprocessing.export_swinunetr \
  --root "$DICOM_ROOT" \
  --surv_csv "$SURV_CSV" \
  --out_root "$OUT_ROOT" \
  --out_csv "$OUT_CSV" \
  --spacing "$SPACING_X" "$SPACING_Y" "$SPACING_Z" \
  --size "$SIZE_D" "$SIZE_H" "$SIZE_W" \
  --margin_mm "$MARGIN_MM" \
  --hu_min "$HU_MIN" \
  --hu_max "$HU_MAX" \
  "${EXTRA_ARGS[@]}" \
  "$@"
