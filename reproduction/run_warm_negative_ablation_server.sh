#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/autodl-tmp/semantic-id-rec-repro/repo}"
PROJECT="${ROOT}/Mini-HSTU-Reproduction"
ENV_ROOT="${ENV_ROOT:-$(dirname "${ROOT}")}"
PYTHON="${PYTHON:-${ENV_ROOT}/env-fbgemm/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="${ENV_ROOT}/env/bin/python"
fi
RESULT_ROOT="${ROOT}/reproduction/warm_negative_ablation"
SUITE_LOG="${RESULT_ROOT}/suite.log"

mkdir -p "${RESULT_ROOT}"
cd "${PROJECT}"
export PYTHONHASHSEED=42
export PYTHONPATH="${PROJECT}/src"
export PYTHONUNBUFFERED=1

timestamp() { date --iso-8601=seconds; }

run_one() {
  local label="$1"
  local experiment="$2"
  local group_dir="${RESULT_ROOT}/${label}"
  local run_dir="${group_dir}/run"
  local log_file="${group_dir}/train.log"
  local gpu_log="${group_dir}/gpu_usage.csv"
  local exit_file="${group_dir}/exit.txt"
  local success_file="${group_dir}/SUCCESS"
  local start_epoch
  start_epoch="$(date +%s)"
  mkdir -p "${group_dir}"

  if [[ -s "${success_file}" ]]; then
    printf '%s SKIP %s (already complete)\n' "$(timestamp)" "${label}" | tee -a "${SUITE_LOG}"
    return 0
  fi
  if [[ -e "${run_dir}" ]]; then
    printf '%s ERROR %s has an incomplete existing run: %s\n' \
      "$(timestamp)" "${label}" "${run_dir}" | tee -a "${SUITE_LOG}"
    return 20
  fi
  mkdir -p "${run_dir}"

  local -a args=(
    "src/generative_recommenders_pl/scripts/train.py"
    "experiment=${experiment}"
    "seed=42"
    "logger=csv"
    "trainer.accelerator=gpu"
    "trainer.devices=1"
    "+data.num_workers=8"
    "hydra.run.dir=${run_dir}"
  )

  {
    echo "label=${label}"
    echo "experiment=${experiment}"
    echo "start_time=$(timestamp)"
    echo "host=$(hostname)"
    echo "git_commit=$(git rev-parse HEAD)"
    echo "python=$(${PYTHON} --version 2>&1)"
    nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
    printf 'command=CUDA_VISIBLE_DEVICES=0 PYTHONPATH=%q %q ' "${PYTHONPATH}" "${PYTHON}"
    printf '%q ' "${args[@]}"
    printf '\n'
  } >> "${log_file}"

  "${PYTHON}" "${args[@]}" --cfg job --resolve \
    > "${group_dir}/hydra_config_resolved.yaml"

  nvidia-smi \
    --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
    --format=csv,noheader,nounits --loop=10 > "${gpu_log}" 2>&1 &
  local sampler_pid=$!

  set +e
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" "${args[@]}" >> "${log_file}" 2>&1
  local status=$?
  set -e
  kill "${sampler_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true

  local peak_memory=""
  if [[ -s "${gpu_log}" ]]; then
    peak_memory="$(awk -F',' 'BEGIN {m=0} {gsub(/ /,"",$3); if ($3+0>m) m=$3+0} END {print m}' "${gpu_log}")"
  fi
  local best_ckpt=""
  best_ckpt="$(sed -n 's/^.*Best ckpt path: //p' "${log_file}" | tail -1 | sed $'s/\033\\[[0-9;]*m//g' || true)"
  if [[ -z "${best_ckpt}" || ! -s "${best_ckpt}" ]]; then
    best_ckpt="$(find "${run_dir}/checkpoints" -maxdepth 1 -type f -name '*.ckpt' 2>/dev/null | sort | tail -1 || true)"
  fi
  local metrics_csv=""
  metrics_csv="$(find "${run_dir}" -type f -path '*/csv/*/metrics.csv' | sort | tail -1 || true)"
  local audit_json="${run_dir}/cold_negative_sampling_audit.json"

  {
    echo "exit_code=${status}"
    echo "end_time=$(timestamp)"
    echo "wall_time_seconds=$(($(date +%s) - start_epoch))"
    echo "run_dir=${run_dir}"
    echo "metrics_csv=${metrics_csv}"
    echo "best_checkpoint=${best_ckpt}"
    echo "peak_gpu_memory_mib=${peak_memory}"
    echo "audit_json=${audit_json}"
  } | tee "${exit_file}" >> "${log_file}"

  if [[ ${status} -ne 0 ]]; then
    return "${status}"
  fi
  if [[ -z "${best_ckpt}" || ! -s "${best_ckpt}" ]]; then
    echo "Missing best checkpoint" | tee -a "${log_file}"
    return 21
  fi
  if [[ -z "${metrics_csv}" || ! -s "${metrics_csv}" ]]; then
    echo "Missing metrics.csv" | tee -a "${log_file}"
    return 22
  fi
  if [[ ! -s "${audit_json}" ]]; then
    echo "Missing cold-negative audit" | tee -a "${log_file}"
    return 23
  fi
  "${PYTHON}" - "${metrics_csv}" "${audit_json}" <<'PY'
import csv
import json
import sys

metrics_path, audit_path = sys.argv[1:]
with open(metrics_path, newline="") as handle:
    rows = list(csv.DictReader(handle))
if not any(any(key.startswith("test/") and value not in ("", None) for key, value in row.items()) for row in rows):
    raise SystemExit("metrics.csv does not contain final full-test metrics")
with open(audit_path) as handle:
    audit = json.load(handle)
required = {
    "audit_passed": True,
    "cold_in_train_history": 0,
    "cold_in_train_targets": 0,
    "cold_in_negative_pool": 0,
}
for key, expected in required.items():
    if audit.get(key) != expected:
        raise SystemExit(f"audit failure: {key}={audit.get(key)!r}")
if audit.get("cold_in_eval_candidates") != audit.get("num_cold_items"):
    raise SystemExit("audit failure: evaluation candidates do not contain every cold item")
PY

  if ! grep -q 'Best ckpt path:.*ckpt' "${log_file}"; then
    echo "Training succeeded but best-checkpoint full-test evidence is missing" | tee -a "${log_file}"
    return 24
  fi
  timestamp > "${success_file}"
  printf '%s COMPLETE %s\n' "$(timestamp)" "${label}" | tee -a "${SUITE_LOG}"
}

run_one hstu_baseline_cold_warm_neg ml-1m-hstu-cold-warm-neg
run_one hstu_dense_cold_warm_neg ml-1m-hstu-dense-cold-warm-neg
run_one rqvae_hstu_route_a_warm_neg ml-1m-hstu-semantic-cold-warm-neg

"${PYTHON}" "${ROOT}/reproduction/summarize_warm_negative_ablation.py" \
  --root "${ROOT}" --result-root "${RESULT_ROOT}"
printf '%s SUITE_COMPLETE\n' "$(timestamp)" | tee -a "${SUITE_LOG}"
