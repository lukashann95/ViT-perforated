#!/bin/bash
#SBATCH --job-name=vit-small
#SBATCH --account=cis260364p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Paths
REPO="/ocean/projects/cis260364p/shared/ViT-perforated"
IMAGENET=/ocean/datasets/community/imagenet
OUTPUT="/ocean/projects/cis260364p/shared/ViT-perforated/train-small-${SLURM_JOB_ID}"

cd "$REPO"
cp /jet/home/lhann/.env "$REPO/.env"

# Fail before staging data if required paths or the environment are unavailable.
: "${LOCAL:?PSC node-local storage path is not set}"
test -d "$IMAGENET/train"
test -d "$IMAGENET/val"
command -v uv

# Copy ImageNet into node-local storage
mkdir -p "$LOCAL/imagenet"

echo "Copying ImageNet..."
copy_started=$SECONDS
for split in train val; do
    echo "Copying ImageNet $split..."
    split_started=$SECONDS
    rsync -a --info=progress2 --stats --outbuf=L \
        "$IMAGENET/$split" "$LOCAL/imagenet/"
    echo "ImageNet $split copy complete in $((SECONDS - split_started)) seconds."
done

echo "ImageNet copy complete in $((SECONDS - copy_started)) seconds total."

# Configure W&B
mkdir -p "$OUTPUT"
export WANDB_DIR="$OUTPUT"
export WANDB_PROJECT="ViT-perforated"

# Train
uv run python -u main.py "$LOCAL/imagenet" \
    --hidden-dim 384 \
    --num-layers 12 \
    --num-heads 6 \
    --patch-size 16 \
    --batch-size 1024 \
    --accum-freq 2 \
    --amp \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs 90 \
    --workers 5 \
    --logs "$OUTPUT" \
    --name "vit_small_amp_${SLURM_JOB_ID}" \
    --report-to wandb
