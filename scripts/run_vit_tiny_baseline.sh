#!/bin/bash
#SBATCH --job-name=vit-baseline
#SBATCH --account=bjto-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/bjto/lhann/runs/slurm-%j.out
# Baseline benchmark: ViT-Tiny/16 (no PAI), 20% stratified ImageNet-1k subset (TRAIN_FRACTION),
# FP32 with TF32 tensor-core matmuls/convs (--tf32; no effect on GPUs without TF32, e.g. V100),
# single GPU. Effective batch BATCH (default 1024), split into microbatches to fit in memory.
#
# Setup (once):
#   - Repo checked out on the branch with the W&B validation logging fix in main.py (the script
#     checks for it): pai_validation = (step % args.log_steps == 0 or step in args.specified_steps)
#   - Dataset prepared with ./scripts/setup_machine.sh (HF imagenet_1k_resized_256 -> ImageFolder)
#   - .env in the repo with WANDB_API_TOKEN (W&B account must be in the IdlF26Group40 entity)
#
# Run (inside tmux), overriding anything that differs on your machine:
#   IMAGENET=~/data/imagenet ./scripts/run_vit_tiny_baseline.sh
# On Delta, from the repo root (the #SBATCH lines are comments elsewhere; settings via --export):
#   sbatch --export=ALL,ACCUM=1 scripts/run_vit_tiny_baseline.sh
# Machine setup (dataset, workers, credentials, GPU) is in scripts/machine_env.sh.
#
# Settings (environment variables, all optional):
#   REPO        repo path                         (default: this script's repo)
#   IMAGENET    dataset path with train/ and val/ (default: $HOME/data/imagenet; Delta: unpacked
#               to node-local disk, see machine_env.sh)
#   EPOCHS      training epochs                   (default: 90)
#   TRAIN_FRACTION  share of ImageNet's training set (default: 0.2, the stratified subset with
#               --subset-seed 42; 1 = all 1.28M images, epochs then take 5x as many steps)
#   WARMUP      linear LR warmup steps            (default: 2000 x 1024/BATCH; the old full-data
#               8xV100 run vit_tiny_amp_8v100_49682577 used 10000)
#   LOG_EPOCHS  extra checkpoint/validation epochs(default: "30 60")
#   BATCH       effective batch size              (default: 1024)
#   SCALING     how LR follows BATCH from the 1024 baseline (LR 1e-3):
#               none (keep it), linear (x BATCH/1024) or sqrt (x sqrt(BATCH/1024)) (default: none)
#   ACCUM       microbatches per step; 4 = 4x256. Use 8 (8x128) if you hit CUDA OOM. (default: BATCH/256)
#   GPU         which GPU to use                  (default: 0; under Slurm: the assigned one)
#   WORKERS     data-loading workers              (default: CPU cores - 1; Slurm: CPUs - 2)
#   RUN_NAME    run / W&B name; reuse it with --resume to continue a run
#   WANDB_CKPT  1 uploads checkpoints to W&B as model artifact checkpoint-<RUN_NAME>, keeping the
#               'last' (every validation) and 'best' versions; 0 turns it off (default: 1)
#
# Warmup is 2000 steps at batch 1024 and scaled to cover the same epochs at other batch sizes.
# --wd is 1e-4 x BATCH/1024 whatever SCALING is: main.py's AdamW coefficient is --wd/--lr, so
# each step decays the weights by --wd (times the LR schedule's fraction of its peak), and this
# keeps the decay per epoch of the 1024 baseline. Scaling it with the LR would give the sqrt arm
# twice the baseline's decay per epoch.
# Full ImageNet, as the old 8xV100 run (70.06% top-1), on one Delta A40 (~17 h):
#   sbatch --export=ALL,ACCUM=1,TRAIN_FRACTION=1,WARMUP=10000 scripts/run_vit_tiny_baseline.sh
# Batch-256 runs with linear and sqrt LR scaling (run names get _b256_linear / _b256_sqrt):
#   BATCH=256 SCALING=linear ./scripts/run_vit_tiny_baseline.sh
#   BATCH=256 SCALING=sqrt ./scripts/run_vit_tiny_baseline.sh
#
# Resume after a crash:
#   RUN_NAME=<name> ./run_vit_tiny_baseline.sh --resume "<REPO>/logs/<name>/checkpoints/checkpoint.pth.tar"
set -euo pipefail

# sbatch runs a copy of this script, so under Slurm the repo is where sbatch was called from.
REPO="${REPO:-$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" && pwd)}"
test -f "$REPO/scripts/machine_env.sh" || { echo "No $REPO/scripts/machine_env.sh (with sbatch, submit from the repo root)"; exit 1; }
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
WD=$(awk "BEGIN { printf \"%g\", 1e-4 * $BATCH / 1024 }")
WARMUP="${WARMUP:-$(( 2000 * 1024 / BATCH ))}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.2}"
PCT=$(awk "BEGIN { printf \"%g\", 100 * $TRAIN_FRACTION }")
# Name tag only for non-default batch sizes, so batch-1024 run names stay as before.
TAG=
if [ "$BATCH" != 1024 ]; then
    TAG="_b${BATCH}"
    if [ "$SCALING" != none ]; then TAG="${TAG}_${SCALING}"; fi
fi
if [ "$WARMUP" != $(( 2000 * 1024 / BATCH )) ]; then TAG="${TAG}_wu${WARMUP}"; fi
LOG_ROOT="$REPO/logs"
SUBSET_TAG=_subset42
if [ "$PCT" = 100 ]; then SUBSET_TAG=; fi
RUN_NAME="${RUN_NAME:-vit_tiny_tf32_1gpu_${PCT}pct_${EPOCHS}ep${TAG}${SUBSET_TAG}_$(hostname -s)_$(date +%Y%m%d_%H%M%S)}"

test -f "$REPO/.env" || { echo "No $REPO/.env with WANDB_API_TOKEN"; exit 1; }
if [ -n "${IMAGENET:-}" ]; then DATA=$IMAGENET; fi
# Sets DATA, WORKERS, PYTHON, the W&B entity/project and the visible GPU.
source "$REPO/scripts/machine_env.sh"

cd "$REPO"

# Fail fast if main.py lacks the W&B fix (otherwise periodic val/* metrics are dropped).
grep -q "pai_validation = (step %" main.py \
    || { echo "main.py: missing the W&B validation logging fix"; exit 1; }

CKPT_ARGS=()
if [ "${WANDB_CKPT:-1}" = 1 ]; then CKPT_ARGS=(--wandb-checkpoints); fi

export WANDB_DIR="$LOG_ROOT/$RUN_NAME"
mkdir -p "$WANDB_DIR"

echo "Run: $RUN_NAME | GPU: ${CUDA_VISIBLE_DEVICES:-all} | data: ${PCT}% | epochs: $EPOCHS | batch: $BATCH | accum: $ACCUM | lr: $LR | wd: $WD | warmup: $WARMUP | workers: $WORKERS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086  # LOG_EPOCHS is a space-separated list on purpose
"$PYTHON" -u main.py "$DATA" \
    --gpu 0 \
    --train-fraction "$TRAIN_FRACTION" \
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
    "${CKPT_ARGS[@]}" \
    "$@" 2>&1 | tee -a "$LOG_ROOT/$RUN_NAME/train.log"
