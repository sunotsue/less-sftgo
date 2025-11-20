#!/usr/bin/env bash
set -euo pipefail


TARGET_TASK_NAMES="${1:-mmlu}"   # default: mmlu
PERCENTAGE="${2:-0.05}"          # default: top 5%

TRAIN_FILE_NAMES="lima"
TRAIN_FILES="/mnt/data/lima.jsonl"
OUTPUT_PATH="/mnt/selected_data/llama3.2-3b-lima-p1.0-lora-seed3"

echo "Running write_selected_data with:"
echo "  TARGET_TASK_NAMES = ${TARGET_TASK_NAMES}"
echo "  TRAIN_FILE_NAMES  = ${TRAIN_FILE_NAMES}"
echo "  TRAIN_FILES       = ${TRAIN_FILES}"
echo "  OUTPUT_PATH       = ${OUTPUT_PATH}"
echo "  PERCENTAGE        = ${PERCENTAGE}"
echo

python -m less.data_selection.write_selected_data \
  --target_task_names "${TARGET_TASK_NAMES}" \
  --train_file_names "${TRAIN_FILE_NAMES}" \
  --train_files ${TRAIN_FILES} \
  --output_path "${OUTPUT_PATH}" \
  --percentage "${PERCENTAGE}"