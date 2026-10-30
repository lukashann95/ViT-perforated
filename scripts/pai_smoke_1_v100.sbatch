#!/bin/bash
#SBATCH --job-name=vit-pai-smoke
#SBATCH --account=cis260364p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=00:30:00
#SBATCH --array=0-1
#SBATCH --output=/ocean/projects/cis260364p/shared/logs/slurm-%A_%a.out

# Smoke test for the PAI code path before the real ablation. Fake data, so no ImageNet copy.
# Switch thresholds of 1 make PAI add a dendrite, switch phases and finish within 30 steps.
#   task 0: control arm (--pai-control)
#   task 1: dendrites on encoder layer 11
# Pass: the log shows "PAI restructured the model" (task 1) and the job exits 0.

set -euo pipefail

# PerforatedAI credentials (PAIEMAIL/PAITOKEN), if this user has them.
if [ -f "$HOME/.pai_env" ]; then source "$HOME/.pai_env"; fi
: "${PAIEMAIL:?PAIEMAIL is not set; create ~/.pai_env}"
: "${PAITOKEN:?PAITOKEN is not set; create ~/.pai_env}"

umask 0002
export OMP_NUM_THREADS=1

ARMS=(control 11)
ARM=${ARMS[$SLURM_ARRAY_TASK_ID]}
if [ "$ARM" = control ]; then
    PAI_ARGS=(--pai-layers 11 --pai-control)
else
    PAI_ARGS=(--pai-layers "$ARM")
fi

REPO="/ocean/projects/cis260364p/${USER}/ViT-perforated"
SHARED=/ocean/projects/cis260364p/shared
BASELINE="$SHARED/checkpoints/vit_tiny_amp_8v100_49682577/model_best.pth.tar"
RUN_NAME="pai_smoke_${ARM}_${SLURM_ARRAY_JOB_ID}"
LOG_ROOT="$SHARED/logs"

cd "$REPO"
test -r "$BASELINE"
command -v uv

uv run python -u main.py \
    --fake-data \
    --gpu 0 \
    --train-fraction 0.2 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size 1024 \
    --accum-freq 2 \
    --amp \
    --mlp-head \
    --lr 1e-4 \
    --max-steps 30 \
    --log-steps 5 \
    --print-freq 5 \
    --workers 5 \
    --pai-init "$BASELINE" \
    "${PAI_ARGS[@]}" \
    --max-dendrites 1 \
    --pai-n-epochs-to-switch 1 \
    --pai-p-epochs-to-switch 1 \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME"
