#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="${PIPELINE_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
MAX_VIDEOS="${MAX_VIDEOS:-999999}"
LOG_FILE="${LOG_FILE:-$PIPELINE_ROOT/logs/workflow_latest.log}"
CONDA_ENV="${CONDA_ENV:-pipeline}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PIPELINE_TMP_DIR="${PIPELINE_TMP_DIR:-$PIPELINE_ROOT/cache/tmp}"
PROCESS_LOG="${PROCESS_LOG:-$PIPELINE_ROOT/logs/process.log}"
PROCESS_CHECK_INTERVAL_SEC="${PROCESS_CHECK_INTERVAL_SEC:-60}"
GPU_SAMPLE_LOG="${GPU_SAMPLE_LOG:-$PIPELINE_ROOT/logs/gpu_samples.log}"
GPU_SAMPLE_INTERVAL_SEC="${GPU_SAMPLE_INTERVAL_SEC:-10}"

mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$PROCESS_LOG")"
mkdir -p "$(dirname "$GPU_SAMPLE_LOG")"
mkdir -p "$PIPELINE_TMP_DIR"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_process() {
  printf '%s %s\n' "$(timestamp)" "$*" >> "$PROCESS_LOG"
}

sample_gpus() {
  while true; do
    {
      printf '%s gpu_summary\n' "$(timestamp)"
      nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
      printf '%s gpu_processes\n' "$(timestamp)"
      nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits
    } >> "$GPU_SAMPLE_LOG" 2>&1
    sleep "$GPU_SAMPLE_INTERVAL_SEC"
  done
}

stop_gpu_sampler() {
  if [[ -n "${gpu_sampler_pid:-}" ]] && kill -0 "$gpu_sampler_pid" 2>/dev/null; then
    kill "$gpu_sampler_pid" 2>/dev/null || true
    wait "$gpu_sampler_pid" 2>/dev/null || true
  fi
}

log_process "status=starting root=$PIPELINE_ROOT workflow=run max_videos=$MAX_VIDEOS conda_env=$CONDA_ENV log=$LOG_FILE"
: > "$LOG_FILE"
: > "$GPU_SAMPLE_LOG"

sample_gpus &
gpu_sampler_pid=$!
trap stop_gpu_sampler EXIT

set +e
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2086
  PYTHONPATH="$PIPELINE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" conda run --no-capture-output -n "$CONDA_ENV" python -m spatial_semantic_pipeline.cli run --max-videos "$MAX_VIDEOS" $EXTRA_ARGS >> "$LOG_FILE" 2>&1 &
else
  PYTHONPATH="$PIPELINE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" conda run --no-capture-output -n "$CONDA_ENV" python -m spatial_semantic_pipeline.cli run --max-videos "$MAX_VIDEOS" >> "$LOG_FILE" 2>&1 &
fi
workflow_pid=$!
set -e

started_at=$(date +%s)
log_process "status=running pid=$workflow_pid"

while kill -0 "$workflow_pid" 2>/dev/null; do
  now=$(date +%s)
  elapsed_sec=$((now - started_at))
  log_bytes=0
  if [[ -f "$LOG_FILE" ]]; then
    log_bytes=$(wc -c < "$LOG_FILE" | tr -d ' ')
  fi
  last_log=""
  if [[ -s "$LOG_FILE" ]]; then
    last_log=$(tail -n 1 "$LOG_FILE" | tr '\t' ' ' | cut -c 1-500)
  fi
  log_process "status=heartbeat pid=$workflow_pid elapsed_sec=$elapsed_sec log_bytes=$log_bytes last_log=${last_log:-none}"
  sleep "$PROCESS_CHECK_INTERVAL_SEC"
done

set +e
wait "$workflow_pid"
exit_code=$?
set -e

finished_at=$(date +%s)
elapsed_sec=$((finished_at - started_at))
log_process "status=finished pid=$workflow_pid exit_code=$exit_code elapsed_sec=$elapsed_sec"
stop_gpu_sampler
trap - EXIT
exit "$exit_code"
