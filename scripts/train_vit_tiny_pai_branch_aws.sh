#!/bin/bash
# PerforatedAI arms branched from one neuron-phase checkpoint, single T4 (AWS g4dn.xlarge).
# PAI_INIT must be a plain checkpoint exported with scripts/export_plain_checkpoint.py from a
# from-scratch PAI run (train_vit_tiny_pai_scratch_aws.sh) before its first switch:
#   PAI_INIT=init.pth.tar ./train_vit_tiny_pai_branch_aws.sh 7        dendrites on layer 7's MLP
#   PAI_INIT=init.pth.tar ./train_vit_tiny_pai_branch_aws.sh control  same wrapping, no dendrites
# Dendrite arms switch into a dendrite phase immediately; the control keeps training the neurons.
# Same data, batch and LR as the from-scratch run. No warmup: the from-scratch run's switch also
# rebuilds the optimizer at the full LR. EPOCHS is only a safety cap; PAI ends the run itself.

set -euo pipefail

ARM=${1:?usage: PAI_INIT=ckpt $0 <layer index|control>}
: "${PAI_INIT:?set PAI_INIT to a plain checkpoint (scripts/export_plain_checkpoint.py)}"
PAI_INIT=$(realpath "$PAI_INIT")
if [ "$ARM" = control ]; then
    PAI_ARGS=(--pai-layers 11 --pai-control)
else
    PAI_ARGS=(--pai-layers "$ARM")
fi

EPOCHS=${EPOCHS:-400}
MAX_DENDRITES=${MAX_DENDRITES:-2}
PAI_TARGET=${PAI_TARGET:-mlp}
TARGET_TAG=""
if [ "$PAI_TARGET" != mlp ]; then TARGET_TAG="_${PAI_TARGET}"; fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
IMAGENET=${IMAGENET:-$HOME/data/imagenet}
LOG_ROOT=${LOG_ROOT:-$HOME/runs}
RUN_NAME="vit_tiny_pai_branch_${ARM}${TARGET_TAG}_20pct_$(date +%Y%m%d_%H%M)"

if [ -f "$HOME/.pai_env" ]; then source "$HOME/.pai_env"; fi
: "${PAIEMAIL:?PAIEMAIL is not set; create ~/.pai_env}"
: "${PAITOKEN:?PAITOKEN is not set; create ~/.pai_env}"

test -d "$IMAGENET/train"
test -d "$IMAGENET/val"
test -r "$PAI_INIT"
mkdir -p "$LOG_ROOT"

export WANDB_DIR="$LOG_ROOT/$RUN_NAME"
export WANDB_ENTITY="IdlF26Group40"
export WANDB_PROJECT="ViT-perforated"
mkdir -p "$WANDB_DIR"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UV=$(command -v uv || echo "$HOME/.local/bin/uv")

cd "$REPO"
"$UV" run python -u main.py "$IMAGENET" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size 512 \
    --accum-freq 1 \
    --amp \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --lr 5e-4 \
    --log-steps 500 \
    --workers 4 \
    --pai-init "$PAI_INIT" \
    "${PAI_ARGS[@]}" \
    --pai-target "$PAI_TARGET" \
    --max-dendrites "$MAX_DENDRITES" \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb
