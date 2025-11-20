
#!/usr/bin/env bash
set -euo pipefail

# Which checkpoint from the warmup run to use (e.g. 500, 1000, 1500...)
CKPT="${1:-500}"

# Name for this training dataset (just used in output path naming)
TRAINING_DATA_NAME="lima"

# Path to the LIMA training file you used in warmup.
# If you used /mnt/data/lima.jsonl in warmup, set that here instead.
TRAINING_DATA_FILE="/mnt/data/lima.jsonl"

# Where your warmup LoRA checkpoints were written.
# Adjust if your warmup script used a different output_dir.
MODEL_DIR="/mnt/llama3.2-3b-lima-p1.0-lora-seed3"

# Gradient type to extract ("adam" is what LESS uses in the README)
GRADIENT_TYPE="sgd" # "adam"

# Dimension of the gradient embedding for the datastore
DIMS="8192"

# --------- DERIVED PATHS (usually no need to edit) ---------

# The specific checkpoint directory to read from
MODEL_PATH="${MODEL_DIR}/checkpoint-${CKPT}"

# Where to put the gradient datastore for this checkpoint + dataset
OUTPUT_PATH="/mnt/grads/llama3.2-3b-lima-p1.0-lora-seed3/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "Using:"
echo "  CKPT              = ${CKPT}"
echo "  TRAINING_DATA_FILE= ${TRAINING_DATA_FILE}"
echo "  MODEL_PATH        = ${MODEL_PATH}"
echo "  OUTPUT_PATH       = ${OUTPUT_PATH}"
echo "  DIMS              = ${DIMS}"
echo "  GRADIENT_TYPE     = ${GRADIENT_TYPE}"
echo

./less/scripts/get_info/grad/get_train_lora_grads.sh \
  "$TRAINING_DATA_FILE" \
  "$MODEL_PATH" \
  "$OUTPUT_PATH" \
  "$DIMS" \
  "$GRADIENT_TYPE"


# ./run_lima_build_grad_store.sh 32