#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/semantic-id-rec-repro/env}"
PYTHON="${ENV_PREFIX}/bin/python"
RECORD_DIR="${ROOT}/reproduction"
LOG_DIR="${RECORD_DIR}/logs"

mkdir -p "${LOG_DIR}"
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}/RQ-VAE-Recommender:${ROOT}/Mini-HSTU-Reproduction/src"

record() {
  local name="$1"
  shift
  local start end status
  start="$(date --iso-8601=seconds)"
  printf '%s\tSTART\t%s\t' "${start}" "${name}" | tee -a "${RECORD_DIR}/commands.log"
  printf '%q ' "$@" | tee -a "${RECORD_DIR}/commands.log"
  printf '\n' | tee -a "${RECORD_DIR}/commands.log"
  set +e
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
  status=${PIPESTATUS[0]}
  set -e
  end="$(date --iso-8601=seconds)"
  printf '%s\tEND\t%s\texit=%s\n' "${end}" "${name}" "${status}" \
    | tee -a "${RECORD_DIR}/commands.log"
  return "${status}"
}

capture_environment() {
  {
    date --iso-8601=seconds
    uname -a
    nvidia-smi
    "${PYTHON}" --version
    "${PYTHON}" -c \
      'import torch; print("torch", torch.__version__); print("cuda", torch.version.cuda); print("cuda_available", torch.cuda.is_available())'
    "${ENV_PREFIX}/bin/pip" freeze
    git -C "${ROOT}" rev-parse HEAD
    git -C "${ROOT}" status --short
  } > "${RECORD_DIR}/environment.txt"
  "${ENV_PREFIX}/bin/pip" freeze > "${RECORD_DIR}/requirements-lock.txt"
}

prepare_data() {
  cd "${ROOT}/Mini-HSTU-Reproduction"
  record prepare_data "${PYTHON}" \
    src/generative_recommenders_pl/scripts/prepare_data.py data=ml-1m
}

train_rqvae() {
  cd "${ROOT}/RQ-VAE-Recommender"
  record train_rqvae "${PYTHON}" train_rqvae.py configs/rqvae_ml1m.gin
  record audit_rqvae "${PYTHON}" scripts/audit_rqvae_quality.py \
    --metrics_csv_path out/rqvae/ml1m/metrics.csv \
    --bridge_path out/rqvae/ml1m/bridge_artifacts.pt \
    --output_json out/rqvae/ml1m/audit_summary.json
  record export_dense "${PYTHON}" scripts/export_dense_features.py \
    --dataset_folder dataset/ml-1m \
    --output_path out/rqvae/ml1m/dense_features.pt \
    --bridge_path out/rqvae/ml1m/bridge_artifacts.pt
}

train_hstu() {
  cd "${ROOT}/Mini-HSTU-Reproduction"
  local experiment="$1"
  record "${experiment}" "${PYTHON}" \
    src/generative_recommenders_pl/scripts/train.py \
    "experiment=${experiment}" trainer=gpu logger=csv
}

case "${1:-all}" in
  env) capture_environment ;;
  prepare) prepare_data ;;
  rqvae) train_rqvae ;;
  baseline) train_hstu ml-1m-hstu-cold ;;
  dense) train_hstu ml-1m-hstu-dense-cold ;;
  route_a) train_hstu ml-1m-hstu-semantic-cold ;;
  route_b) train_hstu ml-1m-hstu-token-cold ;;
  all)
    capture_environment
    prepare_data
    train_rqvae
    train_hstu ml-1m-hstu-cold
    train_hstu ml-1m-hstu-dense-cold
    train_hstu ml-1m-hstu-semantic-cold
    train_hstu ml-1m-hstu-token-cold
    ;;
  *)
    echo "Usage: $0 {env|prepare|rqvae|baseline|dense|route_a|route_b|all}" >&2
    exit 2
    ;;
esac
