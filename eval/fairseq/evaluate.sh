#!/usr/bin/env bash
# Evaluate a fine-tuned Puno Quechua ASR model.
# Run from project root inside Docker.
#
# Usage:
#   bash eval/fairseq/evaluate.sh <checkpoint> [results_dir] [manifest_dir] [subset]
#
# Examples:
#   bash eval/fairseq/evaluate.sh checkpoints/ft_cpt_validated/checkpoint_best.pt results/ft_cpt_validated/test data/manifests/finetune/qxp_v2 test
#   bash eval/fairseq/evaluate.sh checkpoints/ft_cpt_validated/checkpoint_best.pt results/ft_cpt_validated/test_spont data/manifests/finetune/qxp_v2 test_spont

set -euo pipefail

ROOT=$(pwd)
FT_CHECKPOINT=$(realpath "${1:?"Usage: $0 <checkpoint> [results_dir] [manifest_dir] [subset]"}")
RESULTS_DIR=$(realpath --canonicalize-missing "${2:-"$ROOT/results"}")
MANIFEST_DIR=$(realpath "${3:-"$ROOT/data/manifests/finetune/qxp_v2"}")
SUBSET=${4:-"test"}

if [ ! -f "$FT_CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $FT_CHECKPOINT"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "Evaluating: $FT_CHECKPOINT"
echo "  Manifest: $MANIFEST_DIR"
echo "  Subset:   $SUBSET"
echo "  Results:  $RESULTS_DIR"
echo ""

python /workspace/eval/fairseq/infer_patched.py \
    "$MANIFEST_DIR" \
    --gen-subset "$SUBSET" \
    --path "$FT_CHECKPOINT" \
    --results-path "$RESULTS_DIR" \
    --task audio_finetuning \
    --nbest 1 \
    --w2l-decoder viterbi \
    --criterion ctc \
    --labels ltr \
    --max-tokens 5000000 \
    --post-process letter \
    --required-batch-size-multiple 1

echo ""
echo "Done. Results in $RESULTS_DIR"
