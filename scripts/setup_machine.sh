#!/bin/bash
# One-time setup of a fresh GPU machine (tested targets: L40, RTX 3090; also handles RTX 50xx).
# Starts no training run; afterwards use scripts/train_vit_tiny_pai_last6_ln_mlp.sh (or .sbatch).
#
#   ./scripts/setup_machine.sh
#
# On a Slurm cluster, run it on the login node first (it has internet for the downloads), then
# once more inside a GPU job for the smoke test; steps already done are skipped:
#   ./scripts/setup_machine.sh
#   srun --gpus=1 --cpus-per-task=8 --time=1:00:00 -p <partition> --pty ./scripts/setup_machine.sh
#
# Steps (each is skipped when already done, so re-running is safe):
#   1. install uv
#   2. create the venv from uv.lock, with a torch build that has kernels for this GPU
#   3. credentials: create .env from .env.example and check it is filled in
#   4. dataset: download evanarlian/imagenet_1k_resized_256 and unpack it to ImageFolder
#   5. smoke test: GPU, credentials, 6-layer perforation, dataset, and a 30-step PAI run
#
# Settings (environment variables):
#   VENV=path           venv location (default: <repo>/.venv)
#   DATA_DIR=path       dataset root (default: ~/data); ImageFolder ends up in $DATA_DIR/imagenet
#   DOWNLOAD_DATA=0     skip step 4 (e.g. the dataset is already on this machine)
#   PARQUET_DIR=path    unpack parquet shards already on this machine (a folder holding the
#                       train-*/val-*.parquet files) instead of downloading them; never deleted
#   DATA_INCLUDE="..."  download only shards matching these patterns, e.g. for a quick test:
#                       DATA_INCLUDE="data/train-00000-* data/val-*"
#   DELETE_PARQUET=1    delete the ~22 GB of downloaded parquet shards after unpacking
#   SKIP_SMOKE=1        skip step 5

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV=${VENV:-$REPO/.venv}
DATA_DIR=${DATA_DIR:-$HOME/data}
IMAGENET="$DATA_DIR/imagenet"
PARQUET="$DATA_DIR/imagenet_1k_resized_256"
PYTHON="$VENV/bin/python"

step() { echo; echo "##### $* #####"; }
cd "$REPO"

step "1/5 uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

step "2/5 venv at $VENV"
# uv.lock pins torch 2.6.0+cu124, which covers V100/T4/A100/3090/L40 (sm_70-sm_90).
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --frozen
# Empty when no GPU is visible, e.g. on a Slurm login node.
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || true)
if [ -z "$CAP" ]; then
    echo "No GPU visible: keeping the locked torch build (fits the L40 and 3090)"
else
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (compute capability $CAP)"
fi
if [ -n "$CAP" ] && [ "${CAP%%.*}" -ge 10 ]; then
    # Blackwell (RTX 50xx, B200) needs CUDA 12.8 wheels. This overrides the locked torch in this
    # venv only, so run with $VENV/bin/python, not `uv run` (which would restore the lock).
    echo "Blackwell GPU: installing torch 2.7.1+cu128 over the locked cu124 build"
    uv pip install --python "$PYTHON" torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu128
fi

step "3/5 credentials (.env)"
if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    echo "Created $REPO/.env. Fill in WANDB_API_TOKEN, PAIEMAIL and PAITOKEN, then re-run this script."
    exit 1
fi
set -a; source .env; set +a
for var in WANDB_API_TOKEN PAIEMAIL PAITOKEN; do
    if [ -z "${!var:-}" ]; then
        echo "$var is empty in $REPO/.env; fill it in and re-run." >&2
        exit 1
    fi
done
echo ".env has WANDB_API_TOKEN, PAIEMAIL and PAITOKEN"

step "4/5 dataset at $IMAGENET"
if [ -f "$IMAGENET/.unpacked" ]; then
    echo "already unpacked"
elif [ "${DOWNLOAD_DATA:-1}" = 0 ]; then
    echo "DOWNLOAD_DATA=0: skipping"
else
    if [ -n "${PARQUET_DIR:-}" ]; then
        SHARDS="$PARQUET_DIR"
        echo "using the parquet shards in $SHARDS"
    else
        # train + val only (~22 GB); the unlabeled test-* shards are not used. Resumes if interrupted.
        INCLUDE=()
        for pattern in ${DATA_INCLUDE:-data/train-* data/val-*}; do
            INCLUDE+=(--include "$pattern")
        done
        uvx --from huggingface_hub hf download evanarlian/imagenet_1k_resized_256 \
            --repo-type dataset --local-dir "$PARQUET" "${INCLUDE[@]}"
        SHARDS="$PARQUET/data"
    fi
    # Unpack to $IMAGENET/{train,val}/<wnid>/*.JPEG (another ~25 GB), the layout main.py reads.
    uv run --no-project --with pyarrow python scripts/parquet_to_imagefolder.py \
        "$SHARDS" "$IMAGENET" --workers "$(nproc)"
    touch "$IMAGENET/.unpacked"   # marks a complete unpack, so re-runs skip this step
    if [ -z "${PARQUET_DIR:-}" ] && [ "${DELETE_PARQUET:-0}" = 1 ]; then rm -rf "$PARQUET"; fi
fi

step "5/5 smoke test"
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
    echo "SKIP_SMOKE=1: skipping"
    exit 0
fi
if [ -z "$CAP" ]; then
    echo "No GPU visible, so the smoke test cannot run here. On a Slurm cluster, finish with:"
    echo "  srun --gpus=1 --cpus-per-task=8 --time=1:00:00 -p <partition> --pty ./scripts/setup_machine.sh"
    exit 0
fi
DATA_ARGS=()
if [ -d "$IMAGENET/train" ]; then DATA_ARGS=(--data "$IMAGENET"); fi
"$PYTHON" scripts/smoke_test.py --layers 6 7 8 9 10 11 --target ln_mlp ${DATA_ARGS[@]+"${DATA_ARGS[@]}"}

# Short end-to-end PAI run on fake data: thresholds of 1 make PAI switch to a dendrite phase,
# add a dendrite and restructure within 30 steps. Nothing is logged to W&B.
SMOKE_LOGS=$(mktemp -d)
SMOKE_LOG="$SMOKE_LOGS/pai_smoke.log"
echo; echo "30-step PAI training run on fake data (about 5 min; full log: $SMOKE_LOG)"
if ! "$PYTHON" -u main.py --fake-data --gpu 0 --train-fraction 0.2 \
    --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16 --mlp-head \
    --batch-size 64 --accum-freq 1 --tf32 --lr 5e-4 --workers 2 \
    --max-steps 30 --log-steps 5 --print-freq 5 \
    --pai-layers 6 7 8 9 10 11 --pai-target ln_mlp --max-dendrites 1 \
    --pai-n-epochs-to-switch 1 --pai-p-epochs-to-switch 1 --pai-initial-correlation-batches 2 \
    --logs "$SMOKE_LOGS" --name pai_smoke > "$SMOKE_LOG" 2>&1; then
    tail -n 30 "$SMOKE_LOG"
    echo "SMOKE TEST FAILED: the PAI training run crashed (see $SMOKE_LOG)" >&2
    exit 1
fi
# Training progress and PAI phase changes only; the log also has every validation batch.
grep -E '^\[ *[0-9]+/|PAI restructured|Switching back to N|Importing best Model' "$SMOKE_LOG"
if ! grep -q "PAI restructured the model" "$SMOKE_LOG"; then
    echo "SMOKE TEST FAILED: the PAI run never restructured the model (see $SMOKE_LOG)" >&2
    exit 1
fi
echo "PAI training run ok: switched to a dendrite phase and back"
echo; echo "Setup complete. Start a run with scripts/train_vit_tiny_pai_last6_ln_mlp.sh (tmux) or .sbatch (Slurm)"
