#!/usr/bin/env bash
set -u

PID_FILE=${1:?train PID file required}
OUTPUT=${2:?output CSV required}

echo "timestamp,gpu_util_percent,memory_used_mib,memory_total_mib,power_watts" > "${OUTPUT}"
while true; do
  train_pid=$(cat "${PID_FILE}" 2>/dev/null || true)
  if [[ -z "${train_pid}" ]] || ! kill -0 "${train_pid}" 2>/dev/null; then
    break
  fi
  timestamp=$(date --iso-8601=seconds)
  sample=$(nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "${timestamp},${sample}" >> "${OUTPUT}"
  sleep 10
done
