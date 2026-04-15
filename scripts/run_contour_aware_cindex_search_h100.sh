#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_FILE:-$PACKAGE_DIR/scripts/config/contour_aware_cindex_search_h100.env}"

exec env CONFIG_FILE="$CONFIG_PATH" bash "$PACKAGE_DIR/scripts/run_contour_aware_cindex_search.sh" "$@"
