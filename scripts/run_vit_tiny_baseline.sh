#!/bin/bash
# Baseline benchmark: ViT-Tiny/16 (no PAI), 20% stratified ImageNet-1k subset,
# FP32 with TF32 tensor-core matmuls/convs (--tf32; no effect on GPUs without TF32, e.g. V100),
# single GPU. Effective batch 1024, split into microbatches to fit in memory.
#
# Setup (once):
#   - Repo checked out on the branch with the W&B validation logging fix in main.py (the script
#     checks for it): pai_validation = (step % args.log_steps == 0 or step in args.specified_steps)
#   - Dataset prepared with ./scripts/setup_machine.sh (HF imagenet_1k_resized_256 -> ImageFolder)
#   - .env in the repo with WANDB_API_TOKEN (W&B account must be in the IdlF26Group40 entity)
#
# Run (inside tmux), overriding anything that differs on your machine:
#   REPO=~/ViT-perforated IMAGENET=~/data/imagenet ./run_vit_tiny_baseline.sh
#
# Settings (environment variables, all optional):
#   REPO        repo path                         (default: $HOME/Anshu/ViT-perforated)
#   IMAGENET    dataset path with train/ and val/ (default: $HOME/data/imagenet)
#   EPOCHS      training epochs                   (default: 90)
#   LOG_EPOCHS  extra checkpoint/validation epochs(default: "30 60")
#   BATCH       effective batch size              (default: 1024)
#   SCALING     how LR/WD follow BATCH from the 1024 baseline (LR 1e-3, WD 1e-4):
#               none (keep them), linear (x BATCH/1024) or sqrt (x sqrt(BATCH/1024)) (default: none)
#   ACCUM       microbatches per step; 4 = 4x256. Use 8 (8x128) if you hit CUDA OOM. (default: BATCH/256)
#   GPU         which GPU to use                  (default: 0)
#   WORKERS     data-loading workers              (default: CPU cores - 1)
#   RUN_NAME    run / W&B name; reuse it with --resume to continue a run
#
# Warmup is 2000 steps at batch 1024 and scaled to cover the same epochs at other batch sizes.
# Batch-256 runs with linear and sqrt LR/WD scaling (run names get _b256_linear / _b256_sqrt):
#   BATCH=256 SCALING=linear ./scripts/run_vit_tiny_baseline.sh
#   BATCH=256 SCALING=sqrt ./scripts/run_vit_tiny_baseline.sh
#
# Resume after a crash:
#   RUN_NAME=<name> ./run_vit_tiny_baseline.sh --resume "<REPO>/logs/<name>/checkpoints/checkpoint.pth.tar"
set -euo pipefail

REPO="${REPO:-$HOME/Anshu/ViT-perforated}"
IMAGENET="${IMAGENET:-$HOME/data/imagenet}"
EPOCHS="${EPOCHS:-90}"
LOG_EPOCHS="${LOG_EPOCHS:-30 60}"
BATCH="${BATCH:-1024}"
SCALING="${SCALING:-none}"
ACCUM="${ACCUM:-$(( BATCH > 256 ? BATCH / 256 : 1 ))}"
case "$SCALING" in
    none)   SCALE=1 ;;
    linear) SCALE=$(awk "BEGIN { print $BATCH / 1024 }") ;;
    sqrt)   SCALE=$(awk "BEGIN { print sqrt($BATCH / 1024) }") ;;
    *) echo "SCALING must be none, linear or sqrt"; exit 1 ;;
esac
LR=$(awk "BEGIN { printf \"%g\", 1e-3 * $SCALE }")
WD=$(awk "BEGIN { printf \"%g\", 1e-4 * $SCALE }")
WARMUP=$(( 2000 * 1024 / BATCH ))
# Name tag only for non-default batch sizes, so batch-1024 run names stay as before.
TAG=
if [ "$BATCH" != 1024 ]; then
    TAG="_b${BATCH}"
    if [ "$SCALING" != none ]; then TAG="${TAG}_${SCALING}"; fi
fi
LOG_ROOT="$REPO/logs"
RUN_NAME="${RUN_NAME:-vit_tiny_tf32_1gpu_20pct_${EPOCHS}ep${TAG}_subset42_$(hostname -s)_$(date +%Y%m%d_%H%M%S)}"

# Only show one GPU (with 2+ visible GPUs main.py takes its multi-GPU path and crashes).
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS=1
# nproc honours OMP_NUM_THREADS, so unset it to get the real core count.
WORKERS="${WORKERS:-$(( $(env -u OMP_NUM_THREADS nproc) - 1 ))}"

cd "$REPO"
test -d "$IMAGENET/train" || { echo "No $IMAGENET/train"; exit 1; }
test -d "$IMAGENET/val"   || { echo "No $IMAGENET/val"; exit 1; }
command -v uv >/dev/null  || { echo "uv not found"; exit 1; }
test -f .env              || { echo "No $REPO/.env with WANDB_API_TOKEN"; exit 1; }

# Fail fast if main.py lacks the W&B fix (otherwise periodic val/* metrics are dropped).
grep -q "pai_validation = (step %" main.py \
    || { echo "main.py: missing the W&B validation logging fix"; exit 1; }

export WANDB_DIR="$LOG_ROOT/$RUN_NAME"
mkdir -p "$WANDB_DIR"
export WANDB_ENTITY="IdlF26Group40"
export WANDB_PROJECT="ViT-perforated"

echo "Run: $RUN_NAME | GPU: $CUDA_VISIBLE_DEVICES | epochs: $EPOCHS | batch: $BATCH | accum: $ACCUM | lr: $LR | wd: $WD | warmup: $WARMUP | workers: $WORKERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086  # LOG_EPOCHS is a space-separated list on purpose
uv run python -u main.py "$IMAGENET" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size "$BATCH" \
    --accum-freq "$ACCUM" \
    --lr "$LR" \
    --wd "$WD" \
    --tf32 \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --warmup "$WARMUP" \
    --log-epoch $LOG_EPOCHS \
    --workers "$WORKERS" \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb \
    "$@" 2>&1 | tee -a "$LOG_ROOT/$RUN_NAME/train.log"
