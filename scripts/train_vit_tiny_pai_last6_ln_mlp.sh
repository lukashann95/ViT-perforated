#!/bin/bash
# PerforatedAI on ViT-Tiny/16 with dendrites on the last 6 encoder layers (6-11, 0-indexed).
# Each layer's pre-norm ln_2 and its whole MLP (Linear -> GELU -> Linear) form one perforated
# unit (--pai-target ln_mlp): 6 dendrite modules in total. Single GPU; run setup_machine.sh first.
#
# Personal GPU (e.g. a 3090): run inside tmux, a full run takes a day or more:
#   tmux new -s run ./scripts/train_vit_tiny_pai_last6_ln_mlp.sh
# Slurm cluster (e.g. L40): submit the wrapper, which calls this script:
#   sbatch -p <partition> -A <account> scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch
#
# Variants:
#   PAI_INIT=model_best.pth.tar   fine-tune a trained baseline instead of training from scratch
#   CONTROL=1                     same wrapping and schedule, but never add dendrites
#   CONTINUE_RUN=<run name>       continue a run that stopped (e.g. Slurm time limit) from its
#                                 last checkpoint, in the same log directory. PAI can only resume
#                                 a checkpoint saved in a neuron phase. Pass the original's
#                                 PAI_INIT/CONTROL/AMP settings too.
# Extra arguments are passed to main.py, e.g. --max-steps 50 for a quick check.
#
# Defaults follow train_vit_tiny_pai_scratch_aws.sh (fixed 20% subset, batch 512), but in FP32
# with TF32 tensor cores rather than fp16 AMP: about 16 GiB at batch 512 (AMP: about 9), so it
# fits a 24 GB 3090 and a 48 GB L40. Override with environment variables:
#   EPOCHS (150)  BATCH_SIZE (512)  LR (5e-4)  WARMUP (4000 steps, from scratch only)
#   MAX_DENDRITES (2)  LOG_STEPS (500, about one epoch at batch 512)  AMP (0 = FP32 + TF32; 1 = fp16 AMP)
#   WORKERS (Slurm CPUs - 1, else nproc - 1, max 8)  IMAGENET (~/data/imagenet)
#   LOG_ROOT (~/runs)  VENV (<repo>/.venv)  TAG (run-name suffix)

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-$REPO/.venv}
IMAGENET=${IMAGENET:-$HOME/data/imagenet}
LOG_ROOT=${LOG_ROOT:-$HOME/runs}

EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-512}
LR=${LR:-5e-4}
WARMUP=${WARMUP:-4000}
MAX_DENDRITES=${MAX_DENDRITES:-2}
LOG_STEPS=${LOG_STEPS:-500}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
WORKERS=${WORKERS:-$(( CPUS > 9 ? 8 : CPUS - 1 ))}
# Job ID under Slurm, a timestamp otherwise; used to name runs and continuations.
JOB_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M)}

PAI_ARGS=(--pai-layers 6 7 8 9 10 11 --pai-target ln_mlp --max-dendrites "$MAX_DENDRITES")
KIND=scratch
if [ -n "${PAI_INIT:-}" ]; then
    KIND=finetune
elif [ -z "${CONTINUE_RUN:-}" ]; then
    PAI_ARGS+=(--pai-warmup "$WARMUP")
fi
if [ "${CONTROL:-0}" = 1 ]; then
    PAI_ARGS+=(--pai-control)
    KIND="${KIND}_control"
fi
if [ "${AMP:-0}" = 1 ]; then PRECISION=(--amp); else PRECISION=(--tf32); fi

if [ -n "${CONTINUE_RUN:-}" ]; then
    RUN_NAME=$CONTINUE_RUN
    CKPT_DIR="$LOG_ROOT/$RUN_NAME/checkpoints"
    # The continued run overwrites checkpoint.pth.tar at its first validation; keep the start point.
    START_FROM="$CKPT_DIR/continue_${JOB_ID}.pth.tar"
    cp "$CKPT_DIR/checkpoint.pth.tar" "$START_FROM"
    # W&B gets a new run with the original's history copied in. It needs its own W&B dir: the
    # original's wandb-resume.json would make resume='auto' reach for the old run.
    WANDB_RUN="${RUN_NAME}_cont${JOB_ID}"
    START_ARGS=(--resume "$START_FROM" --wandb-name "$WANDB_RUN")
    # Copying history reads the original run from the W&B server, so only when online.
    if [ "${WANDB_MODE:-online}" = online ]; then START_ARGS+=(--wandb-history-from "$RUN_NAME"); fi
    export WANDB_DIR="$LOG_ROOT/$RUN_NAME/$WANDB_RUN"
else
    RUN_NAME="vit_tiny_pai_${KIND}_last6_ln_mlp${TAG:+_$TAG}_20pct_${JOB_ID}"
    START_ARGS=()
    if [ -n "${PAI_INIT:-}" ]; then START_ARGS=(--pai-init "$(realpath "$PAI_INIT")"); fi
    export WANDB_DIR="$LOG_ROOT/$RUN_NAME"
fi

# Credentials: PAIEMAIL/PAITOKEN for PerforatedAI, WANDB_* for logging.
set -a; source "$REPO/.env"; set +a

test -d "$IMAGENET/train"
test -d "$IMAGENET/val"
test -x "$VENV/bin/python"
mkdir -p "$WANDB_DIR"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
echo "Run $RUN_NAME -> $LOG_ROOT/$RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# The venv's python directly, not `uv run`: setup may have installed a GPU-specific torch.
"$VENV/bin/python" -u main.py "$IMAGENET" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size "$BATCH_SIZE" \
    --accum-freq 1 \
    "${PRECISION[@]}" \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --log-steps "$LOG_STEPS" \
    --workers "$WORKERS" \
    "${PAI_ARGS[@]}" \
    ${START_ARGS[@]+"${START_ARGS[@]}"} \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb \
    "$@"
