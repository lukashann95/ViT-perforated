#!/bin/bash
# One ViT-Tiny PAI run: dendrites on encoder block 11 (ln_mlp, Perforated Backpropagation, default
# ImprovedCovarianceDendriteLoss), continued from the step-61000 pre-switch checkpoint of
# vit_tiny_pai_scratch_11_20pct_20261111_0433. Logs to W&B IdlF26Group40/ViT-perforated-plateau.
#   bash scripts/run_pai_l11_from_step61000.sh [GPU]      # GPU defaults to 0
# Optional: WORKERS (default 8), DATA, RESUME, and CONTINUE_RUN=<run name> after a crash.
set -euo pipefail
GPU=${1:-0}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
DATA=${DATA:-$HOME/data/imagenet}                       # ImageFolder with train/<wnid>/ and val/<wnid>/
RESUME=${RESUME:-$(dirname "$REPO")/checkpoints/vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/1111_234822_step61000/checkpoint.pth.tar}

cd "$REPO"
env CUDA_VISIBLE_DEVICES="$GPU" DATA="$DATA" RESUME="$RESUME" \
    HISTORY_FROM=ViT-perforated-amp/vit_tiny_pai_scratch_11_20pct_20261111_0433 \
    WANDB_PROJECT=ViT-perforated-plateau \
    ARM=11 PAI_MODE=pb MAX_DENDRITES=2 EPOCHS=300 \
    BATCH=512 LR=5e-4 LOG_STEPS=500 PAI_PATIENCE=2 PAI_N_EPOCHS_TO_SWITCH=10 \
    WORKERS="${WORKERS:-8}" \
    bash scripts/l40_pai_resume.sbatch
