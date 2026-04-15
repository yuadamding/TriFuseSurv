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
OUT_DIR="${OUT_DIR:-runs/contour_aware_cindex_search_75gb_tf24_followup_${ENDPOINT_LC}}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="30"
MAX_PARALLEL="${MAX_PARALLEL:-0}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-75000}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
FAIL_LOG_TAIL_LINES="${FAIL_LOG_TAIL_LINES:-120}"

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
normalize_list_var TRIALS "tf20_h1095 tf24_anchor_h1095 tf28_h1095 tf24_lrbb2e4_h1095 tf24_locweak_h1095 tf24_capup_h1095"

SUMMARY_CSV="$OUT_DIR/tuning_summary.csv"
SUMMARY_RANKED_CSV="$OUT_DIR/tuning_summary_ranked.csv"
FAIL_CSV="$OUT_DIR/failed_trials.csv"
LOG_DIR="$OUT_DIR/logs"

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
  if (( ${#TRIALS[@]} == 0 )); then
    echo "[error] no trials configured" >&2
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
  echo "[tf24-followup] detected GPUs with >=${MIN_FREE_GPU_MB} MB free: ${AVAILABLE_GPU_IDS[*]} | max_parallel=${#AVAILABLE_GPU_IDS[@]}"
}

append_summary_row() {
  local summary_json="$1"
  local trial="$2"
  python3 - "$summary_json" "$trial" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
trial = sys.argv[2]
summary_csv = Path(sys.argv[3])
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
print(f"[tf24-followup] {trial} weight={row['weight']} OOF c-index={row['c_index']:.4f}")
PY
}

append_failure_row() {
  local trial="$1"
  local fold="$2"
  local log_file="$3"
  local status="$4"
  python3 - "$FAIL_CSV" "$trial" "$fold" "$log_file" "$status" <<'PY'
import csv
import sys
from pathlib import Path

fail_csv = Path(sys.argv[1])
row = {
    "trial": sys.argv[2],
    "fold": int(sys.argv[3]),
    "log_file": sys.argv[4],
    "exit_status": int(sys.argv[5]),
}
write_header = not fail_csv.exists()
with fail_csv.open("a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        w.writeheader()
    w.writerow(row)
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

rows.sort(key=lambda row: float(row["c_index"]), reverse=True)

with ranked_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
    if rows:
      w.writeheader()
      w.writerows(rows)
PY
}

print_failure_log_tail() {
  local log_file="$1"
  local lines="${2:-120}"
  if [[ -f "$log_file" ]]; then
    echo "[tf24-followup] ---- tail -n ${lines} ${log_file} ----" >&2
    tail -n "$lines" "$log_file" >&2 || true
    echo "[tf24-followup] ---- end log tail ----" >&2
  else
    echo "[tf24-followup][warn] missing failure log: $log_file" >&2
  fi
}

configure_trial() {
  local trial="$1"

  local fused_dim="1408"
  local token_dim="1920"
  local token_mlp_hidden_dim="3328"
  local img_proj_hidden_dim="3328"
  local img_tok_ffn_hidden_dim="3328"
  local img_post_hidden_dim="3328"
  local img_attn_heads="8"
  local gate_hidden_dim="1792"
  local rad_hidden_dim="2304"
  local pca="32"
  local time_bin_width="180"
  local risk_horizon="1095"
  local pt_shell_radius="5"
  local ln_shell_radius="5"
  local teacher_force_epochs="24"
  local teacher_force_start="1.0"
  local teacher_force_end="0.0"
  local loc_loss_pt_lambda="0.25"
  local loc_loss_ln_lambda="0.25"
  local loc_presence_lambda="0.05"
  local primary_surv_loss_weight="1.0"
  local aux_surv_loss_weight="0.35"
  local use_multiscale="1"
  local use_radiomics="1"
  local lr_backbone="3e-4"
  local lr_head="8e-5"

  case "$trial" in
    tf20_h1095)
      teacher_force_epochs="20"
      ;;
    tf24_anchor_h1095)
      ;;
    tf28_h1095)
      teacher_force_epochs="28"
      ;;
    tf24_locweak_h1095)
      loc_loss_pt_lambda="0.12"
      loc_loss_ln_lambda="0.12"
      loc_presence_lambda="0.02"
      ;;
    tf24_capup_h1095)
      fused_dim="1536"
      token_dim="2048"
      token_mlp_hidden_dim="3584"
      img_proj_hidden_dim="3584"
      img_tok_ffn_hidden_dim="3584"
      img_post_hidden_dim="3584"
      img_attn_heads="8"
      gate_hidden_dim="2048"
      rad_hidden_dim="2560"
      ;;
    tf24_lrbb2e4_h1095)
      lr_backbone="2e-4"
      lr_head="7e-5"
      ;;
    *)
      echo "[error] unknown tf24 follow-up trial: $trial" >&2
      exit 1
      ;;
  esac

  TRIAL_WRAPPER="$PACKAGE_DIR/scripts/survival/train_contour_aware_survival.sh"
  TRIAL_ARGS=(
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --workers "$WORKERS"
    --no_resume
    --report_metric c_index
    --use_ema
    --use_swa
    --export_extra_risks
    --no_use_checkpoint
    --radiomics_pca_total_components "$pca"
    --time_bin_width_days "$time_bin_width"
    --risk_horizon_days "$risk_horizon"
    --primary_surv_loss_weight "$primary_surv_loss_weight"
    --aux_surv_loss_weight "$aux_surv_loss_weight"
    --ema_decay 0.9995
    --swa_start_epoch 10
    --swa_update_freq_epochs 1
    --pt_shell_radius "$pt_shell_radius"
    --ln_shell_radius "$ln_shell_radius"
    --fused_dim "$fused_dim"
    --img_token_dim "$token_dim"
    --token_mlp_hidden_dim "$token_mlp_hidden_dim"
    --img_proj_hidden_dim "$img_proj_hidden_dim"
    --img_tok_ffn_hidden_dim "$img_tok_ffn_hidden_dim"
    --img_post_hidden_dim "$img_post_hidden_dim"
    --img_attn_heads "$img_attn_heads"
    --gate_hidden_dim "$gate_hidden_dim"
    --rad_hidden_dim "$rad_hidden_dim"
    --lr_backbone "$lr_backbone"
    --lr_head "$lr_head"
    --wd_rad 1e-3
    --modality_dropout_clin_p 0.00
    --modality_dropout_rad_p 0.05
    --clinical_noise_std 0.0
    --radiomics_noise_std 0.0
    --gate_dropout_p 0.05
    --surv_dropout_p 0.10
    --rad_proj_dropout_p 0.05
    --proj_dropout_p 0.10
    --expert_dropout_p 0.00
    --token_mlp_dropout 0.10
    --token_dropout 0.02
    --attn_dropout_p 0.02
    --gate_entropy_lambda 0.001
    --gate_loadbal_lambda 0.001
    --hazard_smooth_lambda 0.001
    --teacher_force_epochs "$teacher_force_epochs"
    --teacher_force_start "$teacher_force_start"
    --teacher_force_end "$teacher_force_end"
    --loc_loss_pt_lambda "$loc_loss_pt_lambda"
    --loc_loss_ln_lambda "$loc_loss_ln_lambda"
    --loc_presence_lambda "$loc_presence_lambda"
    --shell_body_from_ct
  )

  if [[ "$use_multiscale" == "1" ]]; then
    TRIAL_ARGS+=(--use_multiscale)
  fi
  if [[ "$use_radiomics" == "0" ]]; then
    TRIAL_ARGS+=(--no_radiomics)
  fi
}

score_trial_weights() {
  local trial="$1"
  local trial_dir="$OUT_DIR/$trial"
  local weight

  for weight in "${WEIGHTS_TO_SCORE[@]}"; do
    local -a matches=()
    while IFS= read -r path; do
      matches+=("$path")
    done < <(find "$trial_dir" -path "*/test_risks_${weight}.csv" -type f | sort)

    if (( ${#matches[@]} == 0 )); then
      echo "[tf24-followup] trial=$trial weight=$weight -> no matching risk files, skipping"
      continue
    fi
    if (( ${#matches[@]} != ${#FOLDS[@]} )); then
      echo "[tf24-followup] trial=$trial weight=$weight -> found ${#matches[@]} risk files for ${#FOLDS[@]} folds, skipping incomplete OOF score"
      continue
    fi

    local summary_json="$trial_dir/oof_${weight}_summary.json"
    local pred_csv="$trial_dir/oof_${weight}_predictions.csv"
    META_CSV="$META_CSV" \
    ENDPOINT="$ENDPOINT" \
    WEIGHTS="$weight" \
    TRIAL_ROOT="$trial_dir" \
    EXP_PREFIX="$trial" \
    OUT_JSON="$summary_json" \
    OUT_CSV="$pred_csv" \
    bash "$PACKAGE_DIR/scripts/survival/evaluate_oof_cindex.sh"
    append_summary_row "$summary_json" "$trial"
  done
}

run_trial() {
  local trial="$1"
  configure_trial "$trial"

  local trial_dir="$OUT_DIR/$trial"
  rm -rf "$trial_dir"
  mkdir -p "$trial_dir"

  local -a pids=()
  local -a running_gpus=()
  local -a running_folds=()
  local -a running_logs=()
  local -a free_gpus=("${AVAILABLE_GPU_IDS[@]}")
  local total_folds="${#FOLDS[@]}"
  local next_fold_idx=0
  local trial_failed=0

  while (( next_fold_idx < total_folds || ${#pids[@]} > 0 )); do
    while (( trial_failed == 0 && next_fold_idx < total_folds && ${#free_gpus[@]} > 0 )); do
      local fold="${FOLDS[$next_fold_idx]}"
      local gpu="${free_gpus[0]}"
      local exp_name="${trial}_fold$(printf '%02d' "$fold")"
      local log_file="$LOG_DIR/${trial}_fold$(printf '%02d' "$fold").log"
      free_gpus=("${free_gpus[@]:1}")

      echo "[tf24-followup] trial=$trial fold=$fold gpu=$gpu log=$log_file"
      META_CSV="$META_CSV" \
      SPLITS_DIR="$SPLITS_DIR" \
      RADIOMICS_SOURCE="$RADIOMICS_SOURCE" \
      ENDPOINT="$ENDPOINT" \
      OUT_DIR="$trial_dir" \
      EXP_NAME="$exp_name" \
      DEBUG_FOLD="$fold" \
      CUDA_DEVICE="$gpu" \
      DEVICE="$DEVICE" \
      PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
      bash "$TRIAL_WRAPPER" "${TRIAL_ARGS[@]}" >"$log_file" 2>&1 &

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
      echo "[tf24-followup][warn] trial=$trial fold=$TF_WAIT_META2 failed with status=$TF_WAIT_STATUS; see $TF_WAIT_META3" >&2
      append_failure_row "$trial" "$TF_WAIT_META2" "$TF_WAIT_META3" "$TF_WAIT_STATUS"
      print_failure_log_tail "$TF_WAIT_META3" "$FAIL_LOG_TAIL_LINES"
    fi
  done

  if (( trial_failed != 0 )); then
    return 1
  fi

  score_trial_weights "$trial"
}

mkdir -p "$OUT_DIR" "$LOG_DIR"
rm -f "$SUMMARY_CSV" "$SUMMARY_RANKED_CSV" "$FAIL_CSV"
ensure_required_inputs
detect_available_gpus

for trial in "${TRIALS[@]}"; do
  if ! run_trial "$trial"; then
    echo "[tf24-followup][warn] skipping OOF scoring for failed trial: $trial" >&2
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      echo "[tf24-followup][warn] stopping search after first failed trial (STOP_ON_FAILURE=1)." >&2
      break
    fi
  fi
done

write_ranked_summary
echo "[done] contour-aware tf24 follow-up summary -> $SUMMARY_CSV"
if [[ -f "$FAIL_CSV" ]]; then
  echo "[done] failed trials -> $FAIL_CSV"
fi
