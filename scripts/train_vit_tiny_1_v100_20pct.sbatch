#!/bin/bash
#SBATCH --job-name=vit-tiny-20pct
#SBATCH --account=cis260364p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Allow project group members to read and write newly created outputs.
umask 0002

# Avoid CPU thread oversubscription with data-loading workers.
export OMP_NUM_THREADS=1

# Paths
REPO="/ocean/projects/cis260364p/${USER}/ViT-perforated"
IMAGENET=/ocean/datasets/community/imagenet
SHARED=/ocean/projects/cis260364p/shared
RUN_NAME="vit_tiny_amp_1v100_20pct_seed42_${SLURM_JOB_ID}"
LOG_ROOT="$SHARED/logs"
OUTPUT="$LOG_ROOT/$RUN_NAME"
CHECKPOINTS="$SHARED/checkpoints/$RUN_NAME"

mkdir -p "$OUTPUT" "$CHECKPOINTS"
# main.py saves checkpoints under --logs/--name/checkpoints.
ln -sfn "$CHECKPOINTS" "$OUTPUT/checkpoints"

cd "$REPO"

: "${LOCAL:?PSC node-local storage path is not set}"
test -r "$IMAGENET/train.tar"
test -r "$IMAGENET/val.tar"
command -v uv

# Stage the full dataset to preserve ImageFolder ordering across ablations.
# main.py selects a fixed 20% of training examples; validation stays complete.
mkdir -p "$LOCAL/imagenet"

echo "Copying ImageNet..."
copy_started=$SECONDS
for split in train val; do
    echo "Copying ImageNet $split.tar..."
    split_started=$SECONDS
    rsync -a --info=progress2 --stats --outbuf=L \
        "$IMAGENET/$split.tar" "$LOCAL/imagenet/"
    echo "ImageNet $split archive copy complete in $((SECONDS - split_started)) seconds."
    echo "Extracting ImageNet $split (progress every approximately 1 GB)..."
    extract_started=$SECONDS
    tar --extract --file="$LOCAL/imagenet/$split.tar" \
        --directory="$LOCAL/imagenet" --no-same-owner --no-same-permissions \
        --checkpoint=100000 --checkpoint-action=echo
    test -d "$LOCAL/imagenet/$split"
    echo "ImageNet $split extraction complete in $((SECONDS - extract_started)) seconds."
    rm -- "$LOCAL/imagenet/$split.tar"
done

echo "ImageNet staging complete in $((SECONDS - copy_started)) seconds total."

# Config W&B
export WANDB_DIR="$OUTPUT"
export WANDB_ENTITY="IdlF26Group40"
export WANDB_PROJECT="ViT-perforated"

# ViT-Tiny/16: effective batch 1024, split into two microbatches of 512.
# Complete a 30-epoch LR schedule with warmup scaled to the 20% subset.
uv run python -u main.py "$LOCAL/imagenet" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size 1024 \
    --accum-freq 2 \
    --amp \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs 30 \
    --warmup 2000 \
    --workers 5 \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb
