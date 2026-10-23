#!/bin/bash
#SBATCH --job-name=vit-tiny-8v100
#SBATCH --account=cis260364p
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:8
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --output=/ocean/projects/cis260364p/shared/logs/slurm-%j.out

set -euo pipefail

# Allow project group members to read and write newly created outputs.
umask 0002

# Avoid CPU thread oversubscription across eight training processes.
export OMP_NUM_THREADS=1

# Paths
REPO="/ocean/projects/cis260364p/${USER}/ViT-perforated"
IMAGENET=/ocean/datasets/community/imagenet
SHARED=/ocean/projects/cis260364p/shared
RUN_NAME="vit_tiny_amp_8v100_${SLURM_JOB_ID}"
LOG_ROOT="$SHARED/logs"
OUTPUT="$LOG_ROOT/$RUN_NAME"
CHECKPOINTS="$SHARED/checkpoints/$RUN_NAME"

# Create shared/logs before sbatch: Slurm opens its output before this script runs.
mkdir -p "$OUTPUT" "$CHECKPOINTS"
# main.py saves checkpoints under --logs/--name/checkpoints.
ln -sfn "$CHECKPOINTS" "$OUTPUT/checkpoints"

cd "$REPO"

: "${LOCAL:?PSC node-local storage path is not set}"
test -r "$IMAGENET/train.tar"
test -r "$IMAGENET/val.tar"
command -v uv

# Copy ImageNet into node-local storage
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

uv run python -u main.py "$LOCAL/imagenet" \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --dist-url tcp://127.0.0.1:29500 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size 1024 \
    --accum-freq 1 \
    --amp \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs 90 \
    --workers 40 \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb
