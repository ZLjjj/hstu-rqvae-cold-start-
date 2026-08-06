#!/usr/bin/env bash
set -u

ROOT=/root/autodl-tmp/semantic-id-rec-repro/repo
PROJECT=${ROOT}/Mini-HSTU-Reproduction
PYTHON=/root/autodl-tmp/semantic-id-rec-repro/env-fbgemm/bin/python
RECORD_DIR=${ROOT}/reproduction
RUN_ID=$(date +%Y-%m-%d_%H-%M-%S)
RUN_DIR=${PROJECT}/logs/train/runs/full_token_fast_${RUN_ID}
LOG_FILE=${RECORD_DIR}/logs/full_token_route_b_fast_${RUN_ID}.log
GPU_LOG=${RECORD_DIR}/logs/full_token_route_b_fast_${RUN_ID}_gpu.csv
EXIT_FILE=${RECORD_DIR}/full_token_route_b_fast.exit

mkdir -p "${RECORD_DIR}/logs" "${RUN_DIR}"
cd "${PROJECT}"

{
  echo "run_id=${RUN_ID}"
  echo "start_time=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "python=$(${PYTHON} --version 2>&1)"
  "${PYTHON}" -c 'import torch, fbgemm_gpu; print("torch=" + torch.__version__); print("cuda=" + str(torch.version.cuda)); print("fbgemm_ops=" + str(all(hasattr(torch.ops.fbgemm, n) for n in ("asynchronous_complete_cumsum", "dense_to_jagged", "jagged_to_padded_dense"))))'
  nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
  echo "protocol=batch_size=128,precision=bf16-mixed,num_workers=8,max_epochs=50,min_epochs=10,val_batches=5,early_stopping_patience=5,full_test=true"
} >> "${LOG_FILE}"

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits --loop=10 > "${GPU_LOG}" 2>&1 &
SAMPLER_PID=$!
cleanup() {
  kill "${SAMPLER_PID}" 2>/dev/null || true
  wait "${SAMPLER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONUNBUFFERED=1 "${PYTHON}" \
  src/generative_recommenders_pl/scripts/train.py \
  experiment=ml-1m-hstu-full-token-cold \
  seed=42 \
  logger=csv \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  +trainer.precision=bf16-mixed \
  trainer.min_epochs=10 \
  trainer.max_epochs=50 \
  trainer.limit_val_batches=5 \
  data.batch_size=128 \
  trainer.accumulate_grad_batches=1 \
  +data.num_workers=8 \
  callbacks.early_stopping.patience=5 \
  hydra.run.dir="${RUN_DIR}" \
  >> "${LOG_FILE}" 2>&1
status=$?

{
  echo "exit_code=${status}"
  echo "end_time=$(date --iso-8601=seconds)"
  echo "run_dir=${RUN_DIR}"
  echo "log_file=${LOG_FILE}"
  echo "gpu_log=${GPU_LOG}"
} | tee "${EXIT_FILE}" >> "${LOG_FILE}"

exit "${status}"
