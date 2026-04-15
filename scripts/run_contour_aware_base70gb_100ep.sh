#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv}"
RADIOMICS_SOURCE="${RADIOMICS_SOURCE:-cohort_radiomics_patient_wide.csv}"
ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
SPLITS_DIR="${SPLITS_DIR:-runs/opscc_splits_${ENDPOINT_LC}_seed1}"
OUT_DIR="${OUT_DIR:-runs/contour_aware_base70gb_100ep_${ENDPOINT_LC}}"
EXP_PREFIX="${EXP_PREFIX:-base70gb_${ENDPOINT_LC}}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-100}"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
WORKERS="${WORKERS:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-70000}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
TRIAL_NAME="${TRIAL_NAME:-base70gb}"

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

normalize_list_var FOLDS "0 1 2 3"
normalize_list_var WEIGHTS_TO_SCORE "best ema swa last"

SUMMARY_CSV="$OUT_DIR/tuning_summary.csv"
SUMMARY_RANKED_CSV="$OUT_DIR/tuning_summary_ranked.csv"

ensure_required_inputs() {
  local path
  for path in "$META_CSV" "$RADIOMICS_SOURCE"; do
    if [[ ! -f "$path" ]]; then
      echo "[error] required file not found: $path" >&2
      exit 1
    fi
  done
  if [[ ! -d "$SPLITS_DIR" ]]; then
    echo "[error] splits dir not found: $SPLITS_DIR" >&2
    exit 1
  fi
  if [[ ! "$EPOCHS" =~ ^[0-9]+$ ]] || (( EPOCHS <= 0 )); then
    echo "[error] EPOCHS must be a positive integer, got: $EPOCHS" >&2
    exit 1
  fi
}

detect_available_gpus() {
  local -a ids=()
  if declare -F tf_detect_gpu_ids_by_free_mem >/dev/null 2>&1; then
    mapfile -t ids < <(tf_detect_gpu_ids_by_free_mem "$MIN_FREE_GPU_MB")
  elif command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t ids < <(
      nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' -v min_free="$MIN_FREE_GPU_MB" '{gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2); if ($2+0 >= min_free+0) print $1}'
    )
  else
    mapfile -t ids < <(tf_detect_gpu_ids)
  fi
  if (( ${#ids[@]} == 0 )); then
    echo "[error] no GPUs with at least ${MIN_FREE_GPU_MB} MB free were detected." >&2
    exit 1
  fi

  local want="${MAX_PARALLEL}"
  if [[ ! "$want" =~ ^[0-9]+$ ]] || (( want <= 0 || want > ${#ids[@]} )); then
    want="${#ids[@]}"
  fi
  AVAILABLE_GPU_IDS=("${ids[@]:0:$want}")
  echo "[base70gb] detected GPUs with >=${MIN_FREE_GPU_MB} MB free: ${AVAILABLE_GPU_IDS[*]} | max_parallel=${#AVAILABLE_GPU_IDS[@]}"
}

append_summary_row() {
  local summary_json="$1"
  python3 - "$summary_json" "$SUMMARY_CSV" "$TRIAL_NAME" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
trial = sys.argv[3]
data = json.loads(summary_path.read_text())
row = {
    "trial": trial,
    "weight": data["weights"],
    "c_index": data["c_index"],
    "n_predictions": data["n_predictions"],
    "n_evaluable": data["n_evaluable"],
    "n_risk_files": data["n_risk_files"],
}
write_header = not summary_csv.exists()
with summary_csv.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
print(f"[base70gb] weight={row['weight']} OOF c-index={row['c_index']:.4f}")
PY
}

write_ranked_summary() {
  python3 - "$SUMMARY_CSV" "$SUMMARY_RANKED_CSV" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
ranked_csv = Path(sys.argv[2])
if not summary_csv.exists():
    raise SystemExit(0)

with summary_csv.open(newline="") as f:
    rows = list(csv.DictReader(f))

def sort_key(row):
    try:
        value = float(row["c_index"])
    except Exception:
        return float("-inf")
    return value if value == value else float("-inf")

rows.sort(key=sort_key, reverse=True)

with ranked_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
    if rows:
        w.writeheader()
        w.writerows(rows)
PY
}

score_trial_weights() {
  rm -f "$SUMMARY_CSV" "$SUMMARY_RANKED_CSV"
  local weight
  for weight in "${WEIGHTS_TO_SCORE[@]}"; do
    local -a matches=()
    while IFS= read -r path; do
      matches+=("$path")
    done < <(find "$OUT_DIR" -path "*/test_risks_${weight}.csv" -type f | sort)

    if (( ${#matches[@]} == 0 )); then
      echo "[base70gb] weight=$weight -> no matching risk files, skipping"
      continue
    fi
    if (( ${#matches[@]} != ${#FOLDS[@]} )); then
      echo "[base70gb] weight=$weight -> found ${#matches[@]} risk files for ${#FOLDS[@]} folds, skipping incomplete OOF score"
      continue
    fi

    local summary_json="$OUT_DIR/oof_${weight}.json"
    local pred_csv="$OUT_DIR/oof_${weight}.csv"
    META_CSV="$META_CSV" \
    ENDPOINT="$ENDPOINT" \
    WEIGHTS="$weight" \
    TRIAL_ROOT="$OUT_DIR" \
    EXP_PREFIX="$EXP_PREFIX" \
    OUT_JSON="$summary_json" \
    OUT_CSV="$pred_csv" \
    bash "$PACKAGE_DIR/scripts/survival/evaluate_oof_cindex.sh"
    append_summary_row "$summary_json"
  done
  write_ranked_summary
}

run_fold() {
  local fold="$1"
  local gpu="$2"
  local fold_tag
  fold_tag="$(printf '%02d' "$fold")"
  local exp_name="${EXP_PREFIX}_fold${fold_tag}"
  local log_dir="$OUT_DIR/logs"
  mkdir -p "$log_dir"
  local log_file="$log_dir/${EXP_PREFIX}_fold${fold_tag}.log"

  echo "[base70gb] fold=$fold gpu=$gpu exp=$exp_name log=$log_file"

  META_CSV="$META_CSV" \
  SPLITS_DIR="$SPLITS_DIR" \
  RADIOMICS_SOURCE="$RADIOMICS_SOURCE" \
  ENDPOINT="$ENDPOINT" \
  OUT_DIR="$OUT_DIR" \
  EXP_NAME="$exp_name" \
  DEBUG_FOLD="$fold" \
  CUDA_DEVICE="$gpu" \
  DEVICE="$DEVICE" \
  PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
  bash "$PACKAGE_DIR/scripts/survival/train_contour_aware_survival.sh" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --report_metric c_index \
    --use_ema \
    --use_swa \
    --export_extra_risks \
    --no_use_checkpoint \
    --radiomics_pca_total_components 32 \
    --time_bin_width_days 180 \
    --risk_horizon_days 730 \
    --ema_decay 0.9995 \
    --swa_start_epoch 12 \
    --swa_update_freq_epochs 1 \
    --pt_shell_radius 5 \
    --ln_shell_radius 5 \
    --fused_dim 1408 \
    --img_token_dim 1920 \
    --token_mlp_hidden_dim 3328 \
    --img_proj_hidden_dim 3328 \
    --img_tok_ffn_hidden_dim 3328 \
    --img_post_hidden_dim 3328 \
    --img_attn_heads 8 \
    --gate_hidden_dim 1792 \
    --rad_hidden_dim 2304 \
    --lr_backbone 3e-4 \
    --lr_head 8e-5 \
    --wd_rad 1e-3 \
    --modality_dropout_clin_p 0.00 \
    --modality_dropout_rad_p 0.05 \
    --clinical_noise_std 0.0 \
    --radiomics_noise_std 0.0 \
    --gate_dropout_p 0.05 \
    --surv_dropout_p 0.10 \
    --rad_proj_dropout_p 0.05 \
    --proj_dropout_p 0.10 \
    --expert_dropout_p 0.00 \
    --token_mlp_dropout 0.10 \
    --token_dropout 0.02 \
    --attn_dropout_p 0.02 \
    --gate_entropy_lambda 0.001 \
    --gate_loadbal_lambda 0.001 \
    --hazard_smooth_lambda 0.001 \
    --teacher_force_epochs 16 \
    --teacher_force_start 1.0 \
    --teacher_force_end 0.0 \
    --loc_loss_pt_lambda 0.25 \
    --loc_loss_ln_lambda 0.25 \
    --loc_presence_lambda 0.05 \
    --shell_body_from_ct \
    --use_multiscale \
    >"$log_file" 2>&1
}

ensure_required_inputs
detect_available_gpus
mkdir -p "$OUT_DIR"

declare -a pids=()
declare -a running_gpus=()
declare -a running_folds=()
declare -a running_logs=()
declare -a free_gpus=("${AVAILABLE_GPU_IDS[@]}")
trial_failed=0
next_fold_idx=0
total_folds="${#FOLDS[@]}"

while (( next_fold_idx < total_folds || ${#pids[@]} > 0 )); do
  while (( trial_failed == 0 && next_fold_idx < total_folds && ${#free_gpus[@]} > 0 )); do
    fold="${FOLDS[$next_fold_idx]}"
    gpu="${free_gpus[0]}"
    log_file="$OUT_DIR/logs/base70gb_fold$(printf '%02d' "$fold").log"
    free_gpus=("${free_gpus[@]:1}")

    run_fold "$fold" "$gpu" &
    pids+=("$!")
    running_gpus+=("$gpu")
    running_folds+=("$fold")
    running_logs+=("$log_file")
    next_fold_idx=$(( next_fold_idx + 1 ))
  done

  if (( ${#pids[@]} == 0 )); then
    break
  fi

  tf_wait_for_any_tracked_pid pids running_gpus running_folds running_logs
  free_gpus+=("$TF_WAIT_META1")
  if (( TF_WAIT_STATUS != 0 )); then
    trial_failed=1
    echo "[base70gb][warn] fold=$TF_WAIT_META2 gpu=$TF_WAIT_META1 failed with status=$TF_WAIT_STATUS; see $TF_WAIT_META3" >&2
  fi
done

if (( trial_failed != 0 )); then
  exit 1
fi

score_trial_weights
echo "[done] contour-aware ~70GB 100-epoch run finished -> $OUT_DIR"
