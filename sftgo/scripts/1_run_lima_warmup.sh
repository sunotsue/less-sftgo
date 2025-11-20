#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=/mnt/data/lima.jsonl          
MODEL_PATH=meta-llama/Llama-3.2-3B
PERCENTAGE=1.0               # use 100% of LIMA, there are only ~1k examples
DATA_SEED=3
JOB_NAME=llama3.2-3b-lima-p${PERCENTAGE}-lora-seed${DATA_SEED}


./less/scripts/train/warmup_lora_train.sh \
  "$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$DATA_SEED" "$JOB_NAME"
