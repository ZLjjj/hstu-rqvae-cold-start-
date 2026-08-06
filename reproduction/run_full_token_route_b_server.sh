#!/usr/bin/env bash
set -u

ROOT=/root/autodl-tmp/semantic-id-rec-repro/repo
PROJECT=${ROOT}/Mini-HSTU-Reproduction
PYTHON=/root/autodl-tmp/semantic-id-rec-repro/env/bin/python
RECORD_DIR=${ROOT}/reproduction
LOG_FILE=${RECORD_DIR}/logs/full_token_route_b_train.log
EXIT_FILE=${RECORD_DIR}/full_token_route_b_train.exit

mkdir -p "${RECORD_DIR}/logs"
cd "${PROJECT}"

{
  echo "start_time=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "python=$(${PYTHON} --version 2>&1)"
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
  echo "command=experiment=ml-1m-hstu-full-token-cold seed=42 logger=csv trainer.accelerator=gpu trainer.devices=1"
} >> "${LOG_FILE}"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONUNBUFFERED=1 "${PYTHON}" \
  src/generative_recommenders_pl/scripts/train.py \
  experiment=ml-1m-hstu-full-token-cold \
  seed=42 \
  logger=csv \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  >> "${LOG_FILE}" 2>&1
status=$?

{
  echo "exit_code=${status}"
  echo "end_time=$(date --iso-8601=seconds)"
} | tee "${EXIT_FILE}" >> "${LOG_FILE}"
exit "${status}"

