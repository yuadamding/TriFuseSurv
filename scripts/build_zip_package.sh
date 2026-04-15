#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PACKAGE_NAME="${PACKAGE_NAME:-TriFuseSurv_package}"
TOP_LEVEL_ZIP_PATH="${TOP_LEVEL_ZIP_PATH:-../${PACKAGE_NAME}.zip}"
TMP_DIR="$(mktemp -d)"
STAGE_DIR="$TMP_DIR/$PACKAGE_NAME"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$STAGE_DIR"

copy_items=(
  README.md
  pyproject.toml
  scripts
  src
)
if [[ -d docs ]]; then
  copy_items+=(docs)
fi

mkdir -p "$(dirname "$TOP_LEVEL_ZIP_PATH")"
TOP_LEVEL_ZIP_PATH="$(cd "$(dirname "$TOP_LEVEL_ZIP_PATH")" && pwd)/$(basename "$TOP_LEVEL_ZIP_PATH")"

rsync -a \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  "${copy_items[@]}" \
  "$STAGE_DIR"/

rm -f "$TOP_LEVEL_ZIP_PATH"

(
  cd "$TMP_DIR"
  zip -qr "$TOP_LEVEL_ZIP_PATH" "$PACKAGE_NAME"
)

echo "[done] wrote $TOP_LEVEL_ZIP_PATH"
