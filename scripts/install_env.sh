#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
EXTRA_PIP_PACKAGES="${EXTRA_PIP_PACKAGES:-}"
OPENCV_HEADLESS_SPEC="${OPENCV_HEADLESS_SPEC:-opencv-python-headless==4.10.0.84}"

"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -n "$TORCH_INDEX_URL" ]]; then
  python -m pip install --index-url "$TORCH_INDEX_URL" torch
fi

python -m pip install --upgrade -e .
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless || true
python -m pip install --upgrade --no-cache-dir --force-reinstall "$OPENCV_HEADLESS_SPEC"

python - <<'PY'
import importlib
import sys

required = {
    "einops": "einops",
    "numpy": "numpy",
    "pandas": "pandas",
    "pydicom": "pydicom",
    "SimpleITK": "SimpleITK",
    "rt_utils": "rt-utils",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "monai": "monai",
    "cv2": "opencv-python-headless",
}
missing = []
for mod_name, pkg_name in required.items():
    try:
        importlib.import_module(mod_name)
    except Exception:
        missing.append(pkg_name)

if missing:
    sys.stderr.write(
        "[error] environment is still missing required packages: "
        + ", ".join(missing)
        + "\n"
    )
    sys.stderr.write(
        "[error] try: python -m pip install --upgrade " + " ".join(missing) + "\n"
    )
    raise SystemExit(1)
PY

if [[ -n "$EXTRA_PIP_PACKAGES" ]]; then
  python -m pip install $EXTRA_PIP_PACKAGES
fi

cat <<EOF
[done] environment installed in $VENV_DIR

Activate it with:
  source $VENV_DIR/bin/activate

Then run, for example:
  ./scripts/run_contour_aware_survival_serious.sh
  ./scripts/run_contour_aware_cindex_search_h100.sh
EOF
