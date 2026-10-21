#!/bin/bash
#SBATCH --job-name=vit-tiny-8v100
#SBATCH --account=TODO
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:8
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# Activate environment
module load anaconda3
conda deactivate
conda activate /ocean/projects/cis250019p/mzhang23/TA/envs/IDLF26
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1

# Paths
REPO="$HOME/ViT-perforated"
IMAGENET=/ocean/datasets/community/imagenet
OUTPUT="$PROJECT/vit_perforated"

# Copy ImageNet into node-local storage
mkdir -p "$LOCAL/imagenet"

echo "Copying ImageNet..."
rsync -a "$IMAGENET/train" "$LOCAL/imagenet/"
rsync -a "$IMAGENET/val" "$LOCAL/imagenet/"

echo "ImageNet extraction complete."

# Configure W&B
mkdir -p "$OUTPUT"
export WANDB_DIR="$OUTPUT"
export WANDB_PROJECT="ViT-perforated"

# One Python parent spawns eight GPU workers via main.py.
# Do not launch eight parent processes with srun/torchrun.
# Batch 1024 / 8 GPUs = 128 images per GPU with accum-freq 1.
# If memory is insufficient, use accum-freq 2 (64 images per microbatch).
# Workers 32 is divided across GPUs: four data-loader workers per GPU.
# Train
cd "$REPO"

python -u main.py "$LOCAL/imagenet" \
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
    --mlp-head \
    --torchvision-inception-crop \
    --epochs 90 \
    --workers 32 \
    --logs "$OUTPUT" \
    --name vit_tiny_baseline_8v100 \
    --report-to wandb
