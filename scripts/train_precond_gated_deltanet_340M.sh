#!/bin/bash

# Train 340M Preconditioned Gated DeltaNet on SlimPajama 15B tokens
# Usage: bash scripts/train_precond_gated_deltanet_340M.sh [--output DIR] [--wandb-project NAME] [--ngpu N]

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Verify correct FLA is installed before training
FLA_PATH=$(python -c "import fla; print(fla.__file__)" 2>/dev/null)
if [[ "$FLA_PATH" != *"preconditioned-deltanet"* ]]; then
  echo "ERROR: Wrong FLA path: $FLA_PATH"
  echo "Run: python -m pip install -e ${REPO_ROOT}/3rdparty/flash-linear-attention"
  exit 1
fi
echo "FLA OK: $FLA_PATH"

# Defaults
OUTPUT="${REPO_ROOT}/exp/precond_gated_deltanet_340M"
WANDB_PROJECT_NAME=""
NGPU_ARG=8

while [[ $# -gt 0 ]]; do
  case $1 in
    --output) OUTPUT="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT_NAME="$2"; shift 2 ;;
    --ngpu) NGPU_ARG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [ -n "$WANDB_PROJECT_NAME" ]; then
  export WANDB_PROJECT="$WANDB_PROJECT_NAME"
fi

cd "${REPO_ROOT}/3rdparty/flame"

export NNODE=${NNODE:-1}
export NGPU=${NGPU:-$NGPU_ARG}
export LOG_RANK=${LOG_RANK:-0}

bash train.sh \
  --job.config_file flame/models/fla.toml \
  --job.dump_folder "$OUTPUT" \
  --model.config "${REPO_ROOT}/configs/precond_gated_deltanet_340M.json" \
  --model.tokenizer_path fla-hub/transformer-1.3B-100B \
  --training.batch_size 16 \
  --training.seq_len 2048 \
  --training.context_len 2048 \
  --training.gradient_accumulation_steps 2 \
  --training.steps 30000 \
  --training.max_norm 1.0 \
  --training.skip_nan_inf \
  --training.data_parallel_replicate_degree "$NGPU" \
  --training.data_parallel_shard_degree 1 \
  --training.tensor_parallel_degree 1 \
  --training.dataset gmongaras/SlimPajama-627B_Reupload \
  --training.dataset_split train \
  --training.streaming \
  --training.num_workers 16 \
  --training.prefetch_factor 2 \
  --training.seed 42 \
  --training.compile \
  --optimizer.name AdamW \
  --optimizer.lr 4e-4 \
  --optimizer.eps 1e-8 \
  --optimizer.weight_decay 0.01 \
  --lr_scheduler.decay_type cosine \
  --lr_scheduler.warmup_steps 1024 \
  --lr_scheduler.lr_min 0.1 \
  --checkpoint.interval 3000 \
  --checkpoint.load_step 0 \
  --metrics.log_freq 10
