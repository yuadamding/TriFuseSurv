#!/usr/bin/env bash

# Shared GPU detection helpers for TriFuseSurv shell scripts.

tf_detect_gpu_ids() {
  local -a ids=()
  local visible="${CUDA_VISIBLE_DEVICES:-}"
  local item

  if [[ -n "$visible" && "$visible" != "NoDevFiles" ]]; then
    IFS=',' read -r -a ids <<<"$visible"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    while IFS= read -r item; do
      item="$(printf '%s' "$item" | tr -d '[:space:]')"
      [[ -n "$item" ]] && ids+=("$item")
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
  elif command -v python3 >/dev/null 2>&1; then
    while IFS= read -r item; do
      item="$(printf '%s' "$item" | tr -d '[:space:]')"
      [[ -n "$item" ]] && ids+=("$item")
    done < <(python3 - <<'PY'
try:
    import torch
    for i in range(int(torch.cuda.device_count())):
        print(i)
except Exception:
    pass
PY
)
  fi

  for item in "${ids[@]}"; do
    item="$(printf '%s' "$item" | tr -d '[:space:]')"
    [[ -n "$item" ]] && printf '%s\n' "$item"
  done
}

tf_first_gpu_id() {
  local -a ids=()
  mapfile -t ids < <(tf_detect_gpu_ids)
  if (( ${#ids[@]} == 0 )); then
    return 1
  fi
  printf '%s\n' "${ids[0]}"
}

tf_detect_gpu_ids_by_free_mem() {
  local min_free_mb="${1:-0}"
  local -a visible_ids=()
  local -A allowed=()
  local line idx total used free

  mapfile -t visible_ids < <(tf_detect_gpu_ids)
  if (( ${#visible_ids[@]} == 0 )); then
    return 0
  fi
  for idx in "${visible_ids[@]}"; do
    allowed["$idx"]=1
  done

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf '%s\n' "${visible_ids[@]}"
    return 0
  fi

  while IFS=',' read -r idx total used free; do
    idx="$(printf '%s' "$idx" | tr -d '[:space:]')"
    free="$(printf '%s' "$free" | tr -d '[:space:]')"
    [[ -n "$idx" && -n "$free" ]] || continue
    [[ -n "${allowed[$idx]:-}" ]] || continue
    if [[ "$free" =~ ^[0-9]+$ ]] && (( free >= min_free_mb )); then
      printf '%s,%s\n' "$free" "$idx"
    fi
  done < <(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null || true) \
    | sort -t, -k1,1nr \
    | cut -d, -f2
}

tf_wait_for_any_tracked_pid() {
  local -n _pids_ref="$1"
  local -n _meta1_ref="$2"
  local -n _meta2_ref="$3"
  local -n _meta3_ref="$4"
  local _sleep_secs="${5:-1}"
  local _idx _pid _status

  TF_WAIT_PID=""
  TF_WAIT_STATUS=""
  TF_WAIT_META1=""
  TF_WAIT_META2=""
  TF_WAIT_META3=""

  while (( ${#_pids_ref[@]} > 0 )); do
    for _idx in "${!_pids_ref[@]}"; do
      _pid="${_pids_ref[$_idx]}"
      if kill -0 "$_pid" 2>/dev/null; then
        continue
      fi

      if wait "$_pid"; then
        _status=0
      else
        _status=$?
      fi

      TF_WAIT_PID="$_pid"
      TF_WAIT_STATUS="$_status"
      TF_WAIT_META1="${_meta1_ref[$_idx]:-}"
      TF_WAIT_META2="${_meta2_ref[$_idx]:-}"
      TF_WAIT_META3="${_meta3_ref[$_idx]:-}"

      unset '_pids_ref[_idx]' '_meta1_ref[_idx]' '_meta2_ref[_idx]' '_meta3_ref[_idx]'
      _pids_ref=("${_pids_ref[@]}")
      _meta1_ref=("${_meta1_ref[@]}")
      _meta2_ref=("${_meta2_ref[@]}")
      _meta3_ref=("${_meta3_ref[@]}")
      return 0
    done
    sleep "$_sleep_secs"
  done

  return 1
}
