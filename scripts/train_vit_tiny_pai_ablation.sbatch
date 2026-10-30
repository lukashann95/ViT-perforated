#!/bin/bash
#SBATCH --job-name=vit-tiny-pai
#SBATCH --account=cis260364p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=/ocean/projects/cis260364p/shared/logs/slurm-%A_%a.out

# PerforatedAI layer ablation on ViT-Tiny/16, one 1-GPU job per array task.
# All tasks fine-tune the same trained baseline on the same fixed 20% subset:
#   task 0: control (PAI tracker and schedule, no dendrites)
#   task 1: dendrites on the MLP of encoder layer 3  (4th block, 1-indexed)
#   task 2: dendrites on the MLP of encoder layer 7  (8th block)
#   task 3: dendrites on the MLP of encoder layer 11 (12th block)
# Override at submission, e.g. sbatch --export=ALL,EPOCHS=20,MAX_DENDRITES=1 ...

set -euo pipefail

# PerforatedAI credentials (PAIEMAIL/PAITOKEN), if this user has them.
if [ -f "$HOME/.pai_env" ]; then source "$HOME/.pai_env"; fi
: "${PAIEMAIL:?PAIEMAIL is not set; create ~/.pai_env}"
: "${PAITOKEN:?PAITOKEN is not set; create ~/.pai_env}"

# Allow project group members to read and write newly created outputs.
umask 0002

# Avoid CPU thread oversubscription with data-loading workers.
export OMP_NUM_THREADS=1

ARMS=(control 3 7 11)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
if [ "$ARM" = control ]; then
    # Wrapping layer 3 keeps the control's PAI bookkeeping identical; no dendrites are added.
    PAI_ARGS=(--pai-layers 3 --pai-control)
else
    PAI_ARGS=(--pai-layers "$ARM")
fi

EPOCHS=${EPOCHS:-40}
MAX_DENDRITES=${MAX_DENDRITES:-2}

# Paths
REPO="/ocean/projects/cis260364p/${USER}/ViT-perforated"
IMAGENET=/ocean/datasets/community/imagenet
SHARED=/ocean/projects/cis260364p/shared
BASELINE="$SHARED/checkpoints/vit_tiny_amp_8v100_49682577/model_best.pth.tar"
RUN_NAME="vit_tiny_pai_${ARM}_20pct_${SLURM_ARRAY_JOB_ID}"
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
test -r "$BASELINE"
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

# ViT-Tiny/16 fine-tune: batch 1024 in one microbatch (PAI does not support gradient accumulation).
# The 20% subset is about 250 steps per epoch; validate (and score for PAI) once per epoch.
# PAI owns the LR schedule (ReduceLROnPlateau), so --warmup is unused.
uv run python -u main.py "$LOCAL/imagenet" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size 1024 \
    --accum-freq 1 \
    --amp \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --lr 1e-4 \
    --log-steps 250 \
    --workers 5 \
    --pai-init "$BASELINE" \
    "${PAI_ARGS[@]}" \
    --max-dendrites "$MAX_DENDRITES" \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb
