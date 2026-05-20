#!/usr/bin/env bash
# Continued Pre-Training (CPT) launcher for Puno Quechua.
# Run from project root inside the Docker container.
#
# Usage:
#   bash training/scripts/run_cpt.sh [num_gpus] [train_subset] [save_dir]
#
# Examples:
#   bash training/scripts/run_cpt.sh 4 "qxp_scripted,qxp_spontaneous"
#   bash training/scripts/run_cpt.sh 1 "qxp_scripted,qxp_spontaneous,collao" checkpoints/cpt_collao
#
# Note: CUDA_VISIBLE_DEVICES is inherited from the host (set by SLURM job scheduler).
# Do not override it here — the host restricts this container to the allocated GPUs.

set -euo pipefail

ROOT=$(pwd)
NUM_GPUS=${1:-1}
TRAIN_SUBSET=${2:-"qxp_scripted,qxp_spontaneous"}
SAVE_DIR=${3:-"checkpoints/cpt"}

LOG_NAME=$(echo "$SAVE_DIR" | tr '/' '_')
LOG="$ROOT/logs/${LOG_NAME}.log"
mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1
echo "Log: $LOG"
echo "Started: $(date)"
echo ""

# Gradient accumulation scales inversely with GPU count to keep effective batch size constant.
# Base: update_freq=16 on 1 GPU → effective batch = 16 × max_tokens
UPDATE_FREQ=$(( 16 / NUM_GPUS ))

CHECKPOINT_PATH="$ROOT/checkpoints/xlsr2_300m.pt"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: XLSR-128 checkpoint not found at $CHECKPOINT_PATH"
    echo "Download it with:"
    echo "  wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -P checkpoints/"
    exit 1
fi

echo "Starting CPT:"
echo "  GPUs:              $NUM_GPUS"
echo "  CUDA_VISIBLE_DEVS: ${CUDA_VISIBLE_DEVICES:-'(not set, using all)'}"
echo "  Train subset:      $TRAIN_SUBSET"
echo "  Save dir:          $SAVE_DIR"
echo "  Update freq:       $UPDATE_FREQ"
echo "  Checkpoint:        $CHECKPOINT_PATH"
echo ""

fairseq-hydra-train \
    --config-dir "$ROOT/training/configs" \
    --config-name w2v2-large-cpt_qxp \
    hydra.run.dir="/tmp/hydra/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
    common.user_dir="$ROOT/training/custom_task" \
    task.data="$ROOT/data/manifests/pretrain/" \
    "dataset.train_subset='$TRAIN_SUBSET'" \
    optimization.update_freq="[$UPDATE_FREQ]" \
    distributed_training.distributed_world_size=$NUM_GPUS \
    checkpoint.save_dir="$ROOT/$SAVE_DIR" \
    checkpoint.finetune_from_model="$CHECKPOINT_PATH"
