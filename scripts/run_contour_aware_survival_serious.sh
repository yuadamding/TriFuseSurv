#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"

SETTINGS_FILE="${1:-${PIPELINE_SETTINGS:-$PACKAGE_DIR/scripts/config/contour_aware_survival_serious.env}}"
if [[ "$SETTINGS_FILE" != /* && ! -f "$SETTINGS_FILE" && -f "$PACKAGE_DIR/$SETTINGS_FILE" ]]; then
  SETTINGS_FILE="$PACKAGE_DIR/$SETTINGS_FILE"
fi
if [[ ! -f "$SETTINGS_FILE" ]]; then
  echo "[error] settings file not found: $SETTINGS_FILE" >&2
  exit 1
fi

ENV_RUN_PREPROCESS="${RUN_PREPROCESS-__TF_UNSET__}"
ENV_RUN_PREPARE_STAGE2="${RUN_PREPARE_STAGE2-__TF_UNSET__}"
ENV_RUN_SPLITS="${RUN_SPLITS-__TF_UNSET__}"
ENV_RUN_STAGE2="${RUN_STAGE2-__TF_UNSET__}"

# shellcheck disable=SC1090
source "$SETTINGS_FILE"

if [[ "$ENV_RUN_PREPROCESS" != "__TF_UNSET__" ]]; then
  RUN_PREPROCESS="$ENV_RUN_PREPROCESS"
fi
if [[ "$ENV_RUN_PREPARE_STAGE2" != "__TF_UNSET__" ]]; then
  RUN_PREPARE_STAGE2="$ENV_RUN_PREPARE_STAGE2"
fi
if [[ "$ENV_RUN_SPLITS" != "__TF_UNSET__" ]]; then
  RUN_SPLITS="$ENV_RUN_SPLITS"
fi
if [[ "$ENV_RUN_STAGE2" != "__TF_UNSET__" ]]; then
  RUN_STAGE2="$ENV_RUN_STAGE2"
fi

cd "$WORKSPACE_ROOT"
export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"

normalize_list_var() {
  local var_name="$1"
  local default_words="$2"
  local decl
  local -a values=()
  if decl="$(declare -p "$var_name" 2>/dev/null)" && [[ "$decl" == declare\ -a* ]]; then
    eval "values=(\"\${${var_name}[@]}\")"
  else
    local raw
    raw="$(eval "printf '%s' \"\${${var_name}:-${default_words}}\"")"
    read -r -a values <<< "$raw"
  fi
  eval "${var_name}=()"
  local item
  for item in "${values[@]}"; do
    eval "${var_name}+=(\"\$item\")"
  done
}

normalize_list_var STAGE2_FOLDS "0 1 2 3"
normalize_list_var STAGE2_GPU_IDS ""

detect_available_gpus() {
  local -a ids=()
  mapfile -t ids < <(tf_detect_gpu_ids)
  if (( ${#ids[@]} == 0 )); then
    echo "[error] no GPUs detected. Set CUDA_VISIBLE_DEVICES or ensure nvidia-smi is available." >&2
    exit 1
  fi
  AVAILABLE_GPU_IDS=("${ids[@]}")
  echo "[pipeline] detected GPUs: ${AVAILABLE_GPU_IDS[*]}"
}

resolve_stage2_gpus() {
  if (( ${#STAGE2_GPU_IDS[@]} > 0 )); then
    RESOLVED_STAGE2_GPU_IDS=("${STAGE2_GPU_IDS[@]}")
    return
  fi
  detect_available_gpus
  RESOLVED_STAGE2_GPU_IDS=("${AVAILABLE_GPU_IDS[@]}")
}

maybe_enable_missing_prereqs() {
  local preprocess_meta="${PREPROCESS_OUT_ROOT}/${PREPROCESS_OUT_CSV}"
  local stage2_meta="${PREPARE_STAGE2_OUT_DIR}/${PREPARE_STAGE2_OUT_CSV}"

  if [[ "$RUN_PREPROCESS" != "1" ]]; then
    if [[ "$RUN_PREPARE_STAGE2" == "1" || "$RUN_SPLITS" == "1" || "$RUN_STAGE2" == "1" ]]; then
      if [[ ! -f "$preprocess_meta" ]]; then
        echo "[pipeline] missing $preprocess_meta, enabling preprocessing"
        RUN_PREPROCESS=1
      fi
    fi
  fi

  if [[ "$RUN_PREPARE_STAGE2" != "1" ]]; then
    if [[ "$RUN_SPLITS" == "1" || "$RUN_STAGE2" == "1" ]]; then
      if [[ ! -f "$stage2_meta" ]]; then
        echo "[pipeline] missing $stage2_meta, enabling survival metafile preparation"
        RUN_PREPARE_STAGE2=1
      fi
    fi
  fi

  if [[ "$RUN_SPLITS" != "1" && "$RUN_STAGE2" == "1" ]]; then
    if [[ ! -d "$SPLITS_OUT_DIR" ]]; then
      echo "[pipeline] missing $SPLITS_OUT_DIR, enabling split generation"
      RUN_SPLITS=1
    fi
  fi
}

run_preprocess() {
  echo "[pipeline] preprocessing -> $PREPROCESS_OUT_ROOT"
  DICOM_ROOT="$DICOM_ROOT" \
  SURV_CSV="$SURV_CSV" \
  OUT_ROOT="$PREPROCESS_OUT_ROOT" \
  OUT_CSV="$PREPROCESS_OUT_CSV" \
  SPACING_X="$PREPROCESS_SPACING_X" \
  SPACING_Y="$PREPROCESS_SPACING_Y" \
  SPACING_Z="$PREPROCESS_SPACING_Z" \
  SIZE_D="$PREPROCESS_SIZE_D" \
  SIZE_H="$PREPROCESS_SIZE_H" \
  SIZE_W="$PREPROCESS_SIZE_W" \
  MARGIN_MM="$PREPROCESS_MARGIN_MM" \
  HU_MIN="$PREPROCESS_HU_MIN" \
  HU_MAX="$PREPROCESS_HU_MAX" \
  RECURSIVE="$PREPROCESS_RECURSIVE" \
  bash "$PACKAGE_DIR/scripts/preprocessing/export_swinunetr.sh" "${PREPROCESS_EXTRA_ARGS[@]}"
}

run_prepare_stage2() {
  echo "[pipeline] preparing survival metafile -> $PREPARE_STAGE2_OUT_DIR/$PREPARE_STAGE2_OUT_CSV"
  BASE_META_CSV="$PREPARE_STAGE2_BASE_META_CSV" \
  SURV_CSV="$PREPARE_STAGE2_SURV_CSV" \
  CLIN_CSV="$PREPARE_STAGE2_CLIN_CSV" \
  RADIO_CSV="$PREPARE_STAGE2_RADIO_CSV" \
  OUT_DIR="$PREPARE_STAGE2_OUT_DIR" \
  OUT_CSV="$PREPARE_STAGE2_OUT_CSV" \
  bash "$PACKAGE_DIR/scripts/preprocessing/prepare_opscc_tabular.sh" "${PREPARE_STAGE2_EXTRA_ARGS[@]}"
}

run_splits() {
  echo "[pipeline] making splits -> $SPLITS_OUT_DIR"
  META_CSV="$SPLITS_META_CSV" \
  QC_REPORT="$SPLITS_QC_REPORT" \
  QC_POLICY="$SPLITS_QC_POLICY" \
  QC_DROP_AIR_GT="$SPLITS_QC_DROP_AIR_GT" \
  ENDPOINT="$SPLITS_ENDPOINT" \
  CV_FOLDS="$SPLITS_CV_FOLDS" \
  VAL_FRAC="$SPLITS_VAL_FRAC" \
  SPLIT_SEED="$SPLITS_SEED" \
  OUT_DIR="$SPLITS_OUT_DIR" \
  bash "$PACKAGE_DIR/scripts/preprocessing/make_cv_splits.sh" "${SPLITS_EXTRA_ARGS[@]}"
}

validate_contour_warm_start() {
  if [[ -n "${CONTOUR_WARMSTART_CKPT:-}" && ! -f "$CONTOUR_WARMSTART_CKPT" ]]; then
    echo "[error] contour-aware warm-start checkpoint not found: $CONTOUR_WARMSTART_CKPT" >&2
    exit 1
  fi
  if [[ -n "${CONTOUR_WARMSTART_DIR:-}" && ! -d "$CONTOUR_WARMSTART_DIR" ]]; then
    echo "[error] contour-aware warm-start directory not found: $CONTOUR_WARMSTART_DIR" >&2
    exit 1
  fi
}

run_stage2_fold() {
  local fold="$1"
  local gpu="$2"
  local exp_name="${STAGE2_EXP_PREFIX}_fold$(printf '%02d' "$fold")"
  local wrapper="$PACKAGE_DIR/scripts/survival/train_contour_aware_survival.sh"
  local -a extra_args=(
    --epochs "$STAGE2_EPOCHS"
    --batch_size "$STAGE2_BATCH_SIZE"
    --workers "$STAGE2_WORKERS"
  )

  if [[ "$STAGE2_USE_LORA" == "1" ]]; then
    wrapper="$PACKAGE_DIR/scripts/survival/train_contour_aware_survival_lora.sh"
  fi
  if [[ "$STAGE2_USE_RESUME" != "1" ]]; then
    extra_args+=(--no_resume)
  fi
  if [[ "${STAGE2_DEBUG_MAX_TRAIN:-0}" -gt 0 ]]; then
    extra_args+=(--debug_max_train "$STAGE2_DEBUG_MAX_TRAIN")
  fi
  if [[ "${STAGE2_DEBUG_MAX_VAL:-0}" -gt 0 ]]; then
    extra_args+=(--debug_max_val "$STAGE2_DEBUG_MAX_VAL")
  fi
  if [[ "${STAGE2_DEBUG_MAX_TEST:-0}" -gt 0 ]]; then
    extra_args+=(--debug_max_test "$STAGE2_DEBUG_MAX_TEST")
  fi
  extra_args+=("${STAGE2_EXTRA_ARGS[@]}")

  echo "[pipeline] stage 2 fold $fold on GPU $gpu -> $exp_name"
  META_CSV="$STAGE2_META_CSV" \
  SPLITS_DIR="$STAGE2_SPLITS_DIR" \
  RADIOMICS_SOURCE="$STAGE2_RADIOMICS_SOURCE" \
  ENDPOINT="${PRIMARY_ENDPOINT:-${SPLITS_ENDPOINT:-OS}}" \
  OUT_DIR="$STAGE2_OUT_DIR" \
  EXP_NAME="$exp_name" \
  DEBUG_FOLD="$fold" \
  CUDA_DEVICE="$gpu" \
  DEVICE="$STAGE2_DEVICE" \
  CONTOUR_WARMSTART_CKPT="${CONTOUR_WARMSTART_CKPT:-}" \
  CONTOUR_WARMSTART_DIR="${CONTOUR_WARMSTART_DIR:-}" \
  CONTOUR_WARMSTART_NAME="${CONTOUR_WARMSTART_NAME:-best.pt}" \
  bash "$wrapper" "${extra_args[@]}"
}

run_stage2() {
  if [[ ! -f "$STAGE2_META_CSV" ]]; then
    echo "[error] stage 2 meta csv not found: $STAGE2_META_CSV" >&2
    exit 1
  fi
  if [[ ! -d "$STAGE2_SPLITS_DIR" ]]; then
    echo "[error] stage 2 splits dir not found: $STAGE2_SPLITS_DIR" >&2
    exit 1
  fi

  validate_contour_warm_start
  resolve_stage2_gpus

  local -a pids=()
  local -a running_gpus=()
  local -a running_folds=()
  local -a running_logs=()
  local -a free_gpus=("${RESOLVED_STAGE2_GPU_IDS[@]}")
  local total_folds="${#STAGE2_FOLDS[@]}"
  local next_fold_idx=0
  local stage2_failed=0

  while (( next_fold_idx < total_folds || ${#pids[@]} > 0 )); do
    while (( stage2_failed == 0 && next_fold_idx < total_folds && ${#free_gpus[@]} > 0 )); do
      local fold="${STAGE2_FOLDS[$next_fold_idx]}"
      local gpu="${free_gpus[0]}"
      free_gpus=("${free_gpus[@]:1}")
      run_stage2_fold "$fold" "$gpu" &
      pids+=("$!")
      running_gpus+=("$gpu")
      running_folds+=("$fold")
      running_logs+=("")
      next_fold_idx=$(( next_fold_idx + 1 ))
    done

    if (( ${#pids[@]} == 0 )); then
      break
    fi

    tf_wait_for_any_tracked_pid pids running_gpus running_folds running_logs
    free_gpus+=("$TF_WAIT_META1")
    if (( TF_WAIT_STATUS != 0 )); then
      stage2_failed=1
      echo "[pipeline][warn] stage 2 fold $TF_WAIT_META2 on GPU $TF_WAIT_META1 failed with status=$TF_WAIT_STATUS" >&2
    fi
  done

  if (( stage2_failed != 0 )); then
    return 1
  fi
}

mkdir -p runs
maybe_enable_missing_prereqs

if [[ "$RUN_PREPROCESS" == "1" ]]; then
  run_preprocess
fi
if [[ "$RUN_PREPARE_STAGE2" == "1" ]]; then
  run_prepare_stage2
fi
if [[ "$RUN_SPLITS" == "1" ]]; then
  run_splits
fi
if [[ "$RUN_STAGE2" == "1" ]]; then
  run_stage2
fi

echo "[done] contour-aware survival pipeline finished"
