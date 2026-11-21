#!/bin/bash
# Baseline benchmark: ViT-Tiny/16 (no PAI), 20% stratified ImageNet-1k subset,
# strict FP32, single GPU. Effective batch 1024, split into microbatches to fit in memory.
#
# Setup (once):
#   - Repo checked out on the branch with the two main.py edits (the script checks for them):
#       1. torch.set_float32_matmul_precision('highest') + torch.backends.cudnn.allow_tf32 = False
#       2. pai_validation = (step % args.log_steps == 0 or step in args.specified_steps)
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
#   ACCUM       microbatches per step; 4 = 4x256. Use 8 (8x128) if you hit CUDA OOM. (default: 4)
#   GPU         which GPU to use                  (default: 0)
#   WORKERS     data-loading workers              (default: CPU cores - 1)
#   RUN_NAME    run / W&B name; reuse it with --resume to continue a run
#
# Resume after a crash:
#   RUN_NAME=<name> ./run_vit_tiny_baseline.sh --resume "<REPO>/logs/<name>/checkpoints/checkpoint.pth.tar"
set -euo pipefail

REPO="${REPO:-$HOME/Anshu/ViT-perforated}"
IMAGENET="${IMAGENET:-$HOME/data/imagenet}"
EPOCHS="${EPOCHS:-90}"
LOG_EPOCHS="${LOG_EPOCHS:-30 60}"
ACCUM="${ACCUM:-4}"
LOG_ROOT="$REPO/logs"
RUN_NAME="${RUN_NAME:-vit_tiny_fp32_1gpu_20pct_${EPOCHS}ep_subset42_$(hostname -s)_$(date +%Y%m%d_%H%M%S)}"

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

# Fail fast if main.py lacks the benchmark edits (otherwise results/logging differ).
grep -q "^torch.set_float32_matmul_precision('highest')" main.py \
    || { echo "main.py: missing set_float32_matmul_precision('highest')"; exit 1; }
grep -q "^torch.backends.cudnn.allow_tf32 = False" main.py \
    || { echo "main.py: missing torch.backends.cudnn.allow_tf32 = False"; exit 1; }
grep -q "pai_validation = (step %" main.py \
    || { echo "main.py: missing the W&B validation logging fix"; exit 1; }

export WANDB_DIR="$LOG_ROOT/$RUN_NAME"
mkdir -p "$WANDB_DIR"
export WANDB_ENTITY="IdlF26Group40"
export WANDB_PROJECT="ViT-perforated"

echo "Run: $RUN_NAME | GPU: $CUDA_VISIBLE_DEVICES | epochs: $EPOCHS | accum: $ACCUM | workers: $WORKERS"
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
    --batch-size 1024 \
    --accum-freq "$ACCUM" \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --warmup 2000 \
    --log-epoch $LOG_EPOCHS \
    --workers "$WORKERS" \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb \
    "$@" 2>&1 | tee -a "$LOG_ROOT/$RUN_NAME/train.log"
