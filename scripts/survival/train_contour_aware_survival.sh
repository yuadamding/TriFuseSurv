#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$PACKAGE_DIR/.." && pwd)"
cd "$WORKSPACE_ROOT"

export PYTHONPATH="$PACKAGE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
source "$PACKAGE_DIR/scripts/lib/gpu_utils.sh"

META_CSV="${META_CSV:-OPSCC_preprocessed_128/cohort_preprocessed_stage2.csv}"
RADIOMICS_SOURCE="${RADIOMICS_SOURCE:-cohort_radiomics_patient_wide.csv}"
ENDPOINT="${ENDPOINT:-OS}"
ENDPOINT_LC="$(printf '%s' "$ENDPOINT" | tr '[:upper:]' '[:lower:]')"
SPLITS_DIR="${SPLITS_DIR:-runs/opscc_splits_${ENDPOINT_LC}_seed1}"
OUT_DIR="${OUT_DIR:-runs/contour_aware_survival_${ENDPOINT_LC}}"
EXP_NAME="${EXP_NAME:-cv4_contour_aware_${ENDPOINT_LC}_fold03}"
CUDA_DEVICE="${CUDA_DEVICE:-auto}"
REQUESTED_DEVICE="${DEVICE:-cuda:0}"
DEBUG_FOLD="${DEBUG_FOLD:-3}"
WORKERS="${WORKERS:-2}"
CONTOUR_WARMSTART_CKPT="${CONTOUR_WARMSTART_CKPT:-${SHARED_SEG_PRETRAIN_CKPT:-}}"
CONTOUR_WARMSTART_DIR="${CONTOUR_WARMSTART_DIR:-${SHARED_SEG_PRETRAIN_DIR:-}}"
CONTOUR_WARMSTART_NAME="${CONTOUR_WARMSTART_NAME:-${SHARED_SEG_PRETRAIN_NAME:-best.pt}}"

if [[ "$CUDA_DEVICE" == "auto" || -z "$CUDA_DEVICE" ]]; then
  if ! CUDA_DEVICE="$(tf_first_gpu_id)"; then
    echo "[error] could not detect an available GPU for contour-aware survival training." >&2
    exit 1
  fi
fi

if [[ -n "$REQUESTED_DEVICE" && "$REQUESTED_DEVICE" != "cuda" && "$REQUESTED_DEVICE" != "cuda:0" ]]; then
  echo "[warn] overriding DEVICE=$REQUESTED_DEVICE to cuda:0 inside CUDA_VISIBLE_DEVICES=$CUDA_DEVICE for single-GPU isolation"
fi
JOB_DEVICE="cuda:0"
LOG_EVERY_BATCHES="${LOG_EVERY_BATCHES:-50}"

extra_args=()
if [[ -n "$CONTOUR_WARMSTART_CKPT" ]]; then
  extra_args+=(--contour_warmstart_ckpt "$CONTOUR_WARMSTART_CKPT")
elif [[ -n "$CONTOUR_WARMSTART_DIR" ]]; then
  extra_args+=(--contour_warmstart_dir "$CONTOUR_WARMSTART_DIR" --contour_warmstart_name "$CONTOUR_WARMSTART_NAME")
else
  extra_args+=(--no_align_swin_cfg_from_contour_warmstart)
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
PYTHONUNBUFFERED=1 \
python3 -u -m trifusesurv.multimodal_survival.train \
  --meta_csv "$META_CSV" \
  --splits_dir "$SPLITS_DIR" \
  --cv_folds 4 \
  --debug_fold "$DEBUG_FOLD" \
  --strict_splits \
  --endpoint "$ENDPOINT" \
  --ct_col ct_out_path \
  --mask_pt_col mask_primary_out_path \
  --mask_ln_col mask_nodal_out_path \
  --out_dir "$OUT_DIR" \
  --exp_name "$EXP_NAME" \
  --img_size 128 256 256 \
  --epochs 60 \
  --batch_size 1 \
  --workers "$WORKERS" \
  --log_every_batches "$LOG_EVERY_BATCHES" \
  --amp \
  --use_checkpoint \
  --device "$JOB_DEVICE" \
  --use_radiomics \
  --radiomics_root "$RADIOMICS_SOURCE" \
  --use_ema \
  --use_swa \
  --export_extra_risks \
  --lr_backbone 3e-4 \
  --wd_backbone 1e-4 \
  --lr_head 1e-4 \
  --wd_rad 2e-3 \
  --modality_dropout_rad_p 0.20 \
  --primary_surv_loss_weight 1.0 \
  --aux_surv_loss_weight 0.35 \
  --ema_decay 0.9995 \
  --swa_start_epoch 10 \
  --swa_update_freq_epochs 1 \
  --pt_shell_radius 5 \
  --ln_shell_radius 5 \
  --radiomics_pca_total_components 100 \
  --img_token_dim 768 \
  --token_mlp_hidden_dim 1536 \
  --img_proj_hidden_dim 1024 \
  --img_tok_ffn_hidden_dim 1024 \
  --img_post_hidden_dim 1024 \
  --img_attn_heads 4 \
  --gate_hidden_dim 512 \
  --rad_hidden_dim 1024 \
  --rad_proj_dropout_p 0.30 \
  --proj_dropout_p 0.35 \
  --expert_dropout_p 0.15 \
  --token_mlp_dropout 0.55 \
  --token_dropout 0.10 \
  --attn_dropout_p 0.15 \
  --teacher_force_epochs 12 \
  --teacher_force_start 1.0 \
  --teacher_force_end 0.0 \
  --loc_loss_pt_lambda 0.25 \
  --loc_loss_ln_lambda 0.25 \
  --loc_presence_lambda 0.05 \
  --loc_bce_weight 0.5 \
  --loc_dice_weight 0.5 \
  --shell_body_from_ct \
  "${extra_args[@]}" \
  "$@"
