#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_lima_eval_grads.sh <CKPT> <TASK>
#
# Examples:
#   ./run_lima_eval_grads.sh 32 bbh
#   ./run_lima_eval_grads.sh 32 tydiqa
#   ./run_lima_eval_grads.sh 32 mmlu
#   ./run_lima_eval_grads.sh 32 lima_eval   # if you added a custom task

CKPT=${1:-32}          # checkpoint number, e.g. 32
TASK=${2:-mmlu}         # evaluation task name (must match get_validation_dataset.py)

# ----- PATHS FOR YOUR SETUP -----
MODEL_PATH=/mnt/llama3.2-3b-lima-p1.0-lora-seed3/checkpoint-${CKPT}
OUTPUT_ROOT=/mnt/grads/llama3.2-3b-lima-p1.0-lora-seed3
OUTPUT_PATH=${OUTPUT_ROOT}/${TASK}-ckpt${CKPT}-sgd

# For BBH/TydiQA/MMLU downloaded into the repo, this is usually ./data or ../data.
# For a custom task that reads from /mnt/data, adjust get_validation_dataset.py accordingly.
DATA_DIR=./data

# We use 8192 as the default projection dimension (can pass multiple if you want).
DIMS="8192"

echo "Using:"
echo "  CKPT              = ${CKPT}"
echo "  TASK              = ${TASK}"
echo "  MODEL_PATH        = ${MODEL_PATH}"
echo "  OUTPUT_PATH       = ${OUTPUT_PATH}"
echo "  DATA_DIR          = ${DATA_DIR}"
echo "  DIMS              = ${DIMS}"
echo

# Call the LESS helper script (already provided by the repo)
./less/scripts/get_info/grad/get_eval_lora_grads.sh \
  "${TASK}" "${DATA_DIR}" "${MODEL_PATH}" "${OUTPUT_PATH}" "${DIMS}"
