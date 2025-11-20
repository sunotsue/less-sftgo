#!/bin/bash
set -euo pipefail

# Source common training args and 'header' (torchrun command, etc.)
source less/scripts/train/base_training_args.sh

data_dir="$1"      # e.g. ./data
model_path="$2"    # e.g. meta-llama/Llama-3.2-3B
percentage="$3"    # e.g. 1.0
data_seed="$4"     # e.g. 3
job_name="$5"      # e.g. llama3.2-3b-lima-p1.0-lora-seed3

# Where to save checkpoints/logs.
# NOTE: this matches the original LESS layout (../out relative to less/)
output_dir=/mnt/${job_name}
mkdir -p "$output_dir"

# ---------- IMPORTANT CHANGE: use ONLY LIMA ----------
train_files=(
  "$data_dir"
)

# Optional: FSDP configs for specific big models.
# You can extend this for Llama-3.2 if you want FSDP as well.
if [[ "$model_path" == "meta-llama/Llama-2-13b-hf" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
elif [[ "$model_path" == "mistralai/Mistral-7B-v0.1" ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
fi

# Build up the training args string
training_args="$base_training_args \
 --model_name_or_path $model_path \
 --output_dir $output_dir \
 --percentage $percentage \
 --data_seed $data_seed \
 --train_files ${train_files[*]}"

# 'header' is usually something like: torchrun --nproc_per_node=... ...
cmd="$header $training_args"

echo "Running:"
echo "  $cmd"
echo

# Run and tee the log
# (tee is outside the quoted args, so it doesn't get mixed into HF arguments)
eval "$cmd" 2>&1 | tee "$output_dir/train.log"
