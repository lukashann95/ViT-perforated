#!/bin/bash
# PerforatedAI from scratch on ViT-Tiny/16, one RTX 4090 (24 GB), on a machine prepared with
# scripts/setup_machine.sh. Same 20% ImageNet subset and model as scripts/delta_pai_resume.sbatch
# with SCRATCH=1, but at batch 256:
#   ./scripts/train_vit_tiny_pai_scratch_4090.sh            # dendrites on layers 6..11
#   ./scripts/train_vit_tiny_pai_scratch_4090.sh 11         # one layer
#   ./scripts/train_vit_tiny_pai_scratch_4090.sh 6-11 control   # layers 6..11, no dendrites
# Run it inside tmux: one run takes a day or more.
#
# Batch 256, no gradient accumulation. LR and weight decay are the 1024-batch values divided by 4
# (LR 1e-3/4 = 2.5e-4, WD 1e-1/4 = 2.5e-2). The 20% subset is about 1000 steps per epoch at
# batch 256, so validation (and PAI scoring) runs every 1000 steps, about once per epoch. The
# warmup is 8 epochs as in the baseline: 8000 steps at batch 256 (it has to finish before PAI's
# earliest switch, 10 validations = 10000 steps).
# Precision: FP32 (with TF32 tensor cores for matmuls), never fp16 mixed precision (--amp).
#
# Data: uses the ImageFolder copy in $IMAGENET if setup_machine.sh unpacked one (UNPACK=1).
# Otherwise it unpacks the parquet shards there once (~25 GB, kept for later runs).
#
# Optional settings (environment variables):
#   LR, WD, BATCH, LOG_STEPS, PAI_WARMUP, EPOCHS (safety cap; PAI ends the run), MAX_DENDRITES,
#   PAI_TARGET, PAI_MODE (pb | gd), TAG (appended to the run name),
#   DATA_DIR, IMAGENET, PARQUET_DIR (same defaults as setup_machine.sh), LOG_ROOT (default ~/runs)

set -euo pipefail

ARM=${1:-6-11}
CONTROL=0
if [ "${2:-}" = control ]; then CONTROL=1; fi

BATCH=${BATCH:-256}
LR=${LR:-2.5e-4}
WD=${WD:-2.5e-2}
LOG_STEPS=${LOG_STEPS:-1000}
PAI_WARMUP=${PAI_WARMUP:-8000}
EPOCHS=${EPOCHS:-300}
MAX_DENDRITES=${MAX_DENDRITES:-2}
PAI_TARGET=${PAI_TARGET:-ln_mlp}
PAI_MODE=${PAI_MODE:-pb}

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${VENV:-$REPO/.venv}/bin/python
DATA_PYTHON=${DATA_VENV:-$REPO/.venv-data}/bin/python
DATA_DIR=${DATA_DIR:-$(dirname "$REPO")}
PARQUET_DIR=${PARQUET_DIR:-$DATA_DIR/imagenet_1k_resized_256/data}
IMAGENET=${IMAGENET:-$DATA_DIR/imagenet}
LOG_ROOT=${LOG_ROOT:-$HOME/runs}

# ARM: one layer (11) or an inclusive range (6-11).
if [[ $ARM =~ ^([0-9]+)-([0-9]+)$ ]]; then
    LAYERS=($(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"))
    [ ${#LAYERS[@]} -gt 0 ] || { echo "ARM=$ARM is an empty range" >&2; exit 1; }
elif [[ $ARM =~ ^[0-9]+$ ]]; then
    LAYERS=("$ARM")
else
    echo "usage: $0 [layer | first-last] [control]   (default: 6-11)" >&2; exit 1
fi
PAI_ARGS=(--pai-layers "${LAYERS[@]}")
ARM_TAG=$ARM
if [ "$CONTROL" = 1 ]; then PAI_ARGS+=(--pai-control); ARM_TAG=${ARM}_control; fi
MODE_ARGS=()
case "$PAI_MODE" in
    pb) ;;
    gd) MODE_ARGS=(--pai-gd) ;;
    *) echo "PAI_MODE must be pb or gd" >&2; exit 1 ;;
esac
PRECISION=(--tf32)
PREC_TAG=fp32
RUN_NAME="vit_tiny_pai_scratch_${ARM_TAG}_${PAI_TARGET}_${PAI_MODE}_${PREC_TAG}_b${BATCH}${TAG:+_$TAG}_20pct_$(date +%Y%m%d_%H%M)"

# Credentials (WANDB_API_TOKEN, PAIEMAIL, PAITOKEN) from the .env that setup_machine.sh checked.
[ -f "$REPO/.env" ] || { echo "no $REPO/.env; run scripts/setup_machine.sh first" >&2; exit 1; }
set -a; source "$REPO/.env"; set +a
: "${PAIEMAIL:?PAIEMAIL is not set in $REPO/.env}"
: "${PAITOKEN:?PAITOKEN is not set in $REPO/.env}"
[ -x "$PYTHON" ] || { echo "no training venv at $PYTHON; run scripts/setup_machine.sh first" >&2; exit 1; }

# Dataset: unpack the shards once to $IMAGENET if setup_machine.sh did not (UNPACK=1).
if [ ! -f "$IMAGENET/.unpacked" ]; then
    ls "$PARQUET_DIR"/train-*.parquet >/dev/null 2>&1 \
        || { echo "no unpacked data in $IMAGENET and no shards in $PARQUET_DIR" >&2; exit 1; }
    mkdir -p "$IMAGENET"
    avail_gb=$(df -BG --output=avail "$IMAGENET" | tail -1 | tr -dc 0-9)
    if [ "$avail_gb" -lt 40 ]; then
        echo "only ${avail_gb} GB free for $IMAGENET; need ~30 GB" >&2; exit 1
    fi
    echo "Unpacking ImageNet to $IMAGENET (one time, $avail_gb GB free)..."
    start=$SECONDS
    "$DATA_PYTHON" "$REPO/scripts/parquet_to_imagefolder.py" "$PARQUET_DIR" "$IMAGENET" --workers "$(nproc)"
    touch "$IMAGENET/.unpacked"
    echo "Unpacked in $((SECONDS - start)) s"
fi
test -d "$IMAGENET/train"
test -d "$IMAGENET/val"

# PAI runs refuse to start when more than one GPU is visible: use the one with the most free
# memory unless CUDA_VISIBLE_DEVICES already picks one.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
else
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1)
fi
export CUDA_VISIBLE_DEVICES

mkdir -p "$LOG_ROOT"
export WANDB_DIR="$LOG_ROOT/$RUN_NAME"
export WANDB_ENTITY=${WANDB_ENTITY:-IdlF26Group40}
export WANDB_PROJECT=${WANDB_PROJECT:-ViT-perforated}
mkdir -p "$WANDB_DIR"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKERS=${WORKERS:-$(( $(nproc) > 4 ? $(nproc) - 2 : 2 ))}
echo "Run $RUN_NAME: layers ${LAYERS[*]}, batch $BATCH, lr $LR, wd $WD, log every $LOG_STEPS steps"
echo "GPU $CUDA_VISIBLE_DEVICES: $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total --format=csv,noheader)"
echo "Logs: $LOG_ROOT/$RUN_NAME"

cd "$REPO"
"$PYTHON" -u main.py "$IMAGENET" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size "$BATCH" \
    --accum-freq 1 \
    "${PRECISION[@]}" \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --wd "$WD" \
    --pai-warmup "$PAI_WARMUP" \
    --log-steps "$LOG_STEPS" \
    --workers "$WORKERS" \
    "${PAI_ARGS[@]}" \
    ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} \
    --pai-target "$PAI_TARGET" \
    --max-dendrites "$MAX_DENDRITES" \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb
