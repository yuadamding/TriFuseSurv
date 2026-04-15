#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
OUT_DIR_DEFAULT="runs/contour_aware_v75_tri_h1095_tf24_4fold_${ENDPOINT_LC}"

exec env \
  ENDPOINT="$ENDPOINT" \
  FOLDS="0 1 2 3" \
  TRIALS="v75_tri_h1095_tf24" \
  OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}" \
  bash "$PACKAGE_DIR/scripts/run_contour_aware_cindex_search_75gb_30ep.sh" "$@"
