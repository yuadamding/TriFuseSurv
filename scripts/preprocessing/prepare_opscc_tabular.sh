#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

BASE_META_CSV="${BASE_META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed.csv}"
SURV_CSV="${SURV_CSV:-opscc_survival_time_event.csv}"
CLIN_CSV="${CLIN_CSV:-clinical_covariate.csv}"
RADIO_CSV="${RADIO_CSV:-cohort_radiomics_patient_wide.csv}"
OUT_DIR="${OUT_DIR:-OPSCC_preprocessed_128}"
OUT_CSV="${OUT_CSV:-cohort_preprocessed_stage2.csv}"

python3 -m trifusesurv.preprocessing.prepare_opscc_tabular \
  --base_meta_csv "$BASE_META_CSV" \
  --surv_csv "$SURV_CSV" \
  --clin_csv "$CLIN_CSV" \
  --radio_csv "$RADIO_CSV" \
  --out_dir "$OUT_DIR" \
  --out_csv "$OUT_CSV" \
  "$@"
