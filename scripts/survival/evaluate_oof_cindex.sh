#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv}"
ENDPOINT="${ENDPOINT:-OS}"
WEIGHTS="${WEIGHTS:-ema}"
TRIAL_ROOT="${TRIAL_ROOT:-}"
EXP_PREFIX="${EXP_PREFIX:-}"
OUT_JSON="${OUT_JSON:-}"
OUT_CSV="${OUT_CSV:-}"

ARGS=(
  --meta_csv "$META_CSV"
  --endpoint "$ENDPOINT"
  --weights "$WEIGHTS"
)

if [[ -n "$TRIAL_ROOT" ]]; then
  ARGS+=(--trial_root "$TRIAL_ROOT")
fi
if [[ -n "$EXP_PREFIX" ]]; then
  ARGS+=(--exp_prefix "$EXP_PREFIX")
fi
if [[ -n "$OUT_JSON" ]]; then
  ARGS+=(--out_json "$OUT_JSON")
fi
if [[ -n "$OUT_CSV" ]]; then
  ARGS+=(--out_csv "$OUT_CSV")
fi

python3 -m trifusesurv.multimodal_survival.evaluate_oof_cindex \
  "${ARGS[@]}" \
  "$@"
