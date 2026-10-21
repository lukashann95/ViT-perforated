#!/bin/bash
#SBATCH --job-name=vit-small
#SBATCH --account=TODO
#SBATCH --partition=GPU-shared
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Activate environment
module load anaconda3
conda deactivate
conda activate /ocean/projects/cis250019p/mzhang23/TA/envs/IDLF26 && export PYTHONNOUSERSITE=1

# Paths
REPO="$HOME/ViT-perforated"
IMAGENET=/ocean/datasets/community/imagenet
OUTPUT="$PROJECT/vit_perforated"

# Copy ImageNet into node-local storage
mkdir -p "$LOCAL/imagenet"

echo "Copying ImageNet..."
rsync -a $IMAGENET/train "$LOCAL/imagenet/"
rsync -a $IMAGENET/val "$LOCAL/imagenet/"

echo "ImageNet extraction complete."

# Configure W&B
mkdir -p "$OUTPUT"
export WANDB_DIR="$OUTPUT"
export WANDB_PROJECT="ViT-perforated"

# Train
cd "$REPO"

python -u main.py "$LOCAL/imagenet" \
    --hidden-dim 384 \
    --num-layers 12 \
    --num-heads 6 \
    --patch-size 16 \
    --batch-size 1024 \
    --accum-freq 16 \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs 90 \
    --workers 8 \
    --logs "$OUTPUT" \
    --name vit_small_baseline \
    --report-to wandb
