#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_lima_matching_mmlu.sh
#
# Assumes you have already run:
#   1) warmup LoRA training → /mnt/llama3.2-3b-lima-p1.0-lora-seed3/checkpoint-32
#   2) training grads for LIMA:
#      /mnt/grads/llama3.2-3b-lima-p1.0-lora-seed3/lima-ckpt32-sgd/dim8192
#   3) eval grads for MMLU:
#      /mnt/grads/llama3.2-3b-lima-p1.0-lora-seed3/mmlu-ckpt32-sgd/dim8192

DIM=8192

# ---- TRAINING GRADIENTS (LIMA) ----

GRADIENT_PATH="/mnt/grads/llama3.2-3b-lima-p1.0-lora-seed3/{}-ckpt{}-sgd/dim${DIM}"

# You only trained on LIMA for this run
TRAIN_FILE_NAMES="lima"

# You only used checkpoint 32 for building the gradient datastore
CKPTS="32"

# With only 1 checkpoint, the weight can just be 1.0
CHECKPOINT_WEIGHTS="1.0"

# ---- VALIDATION GRADIENTS (MMLU) ----
VALIDATION_GRADIENT_PATH="/mnt/grads/llama3.2-3b-lima-p1.0-lora-seed3/{}-ckpt{}-sgd/dim${DIM}"

# Target task(s) whose validation grads you computed.
TARGET_TASK_NAMES="mmlu"

# Where to dump selected training data indices / metadata
SELECTED_DATA_OUTPUT_PATH="/mnt/selected_data/llama3.2-3b-lima-p1.0-lora-seed3"

echo "Running matching with:"
echo "  DIM                        = ${DIM}"
echo "  GRADIENT_PATH              = ${GRADIENT_PATH}"
echo "  TRAIN_FILE_NAMES           = ${TRAIN_FILE_NAMES}"
echo "  CKPTS                      = ${CKPTS}"
echo "  CHECKPOINT_WEIGHTS         = ${CHECKPOINT_WEIGHTS}"
echo "  VALIDATION_GRADIENT_PATH   = ${VALIDATION_GRADIENT_PATH}"
echo "  TARGET_TASK_NAMES          = ${TARGET_TASK_NAMES}"
echo "  SELECTED_DATA_OUTPUT_PATH  = ${SELECTED_DATA_OUTPUT_PATH}"
echo

./less/scripts/data_selection/matching.sh \
  "${GRADIENT_PATH}" \
  "${TRAIN_FILE_NAMES}" \
  "${CKPTS}" \
  "${CHECKPOINT_WEIGHTS}" \
  "${VALIDATION_GRADIENT_PATH}" \
  "${TARGET_TASK_NAMES}" \
  "${SELECTED_DATA_OUTPUT_PATH}"
