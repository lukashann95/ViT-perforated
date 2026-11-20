#!/bin/bash
#SBATCH --job-name=vit-pai-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out
# Partition, account and GPU type differ per cluster: pass them at submission (-p, -A, and e.g.
# --gres=gpu:l40:1 if the cluster wants the GPU type named), or add #SBATCH lines here.

# L40 version of scripts/delta_pai_resume.sbatch, for a machine set up with
# scripts/setup_machine.sh (default settings: .venv and .venv-data in the repo, parquet shards in
# <repo parent>/imagenet_1k_resized_256/data, keys in <repo>/.env). The shards are unpacked to
# node-local disk at the start of the job. Submit from the repo root:
#   cd ViT-perforated
#   sbatch -p <partition> -A <account> scripts/l40_pai_resume.sbatch          # layers 9, 10 and 11 in one run
#   sbatch -p <partition> -A <account> --export=ALL,ARM=7 scripts/l40_pai_resume.sbatch   # one layer
#   for a in 9 10 11; do sbatch ... --export=ALL,ARM=$a scripts/l40_pai_resume.sbatch; done  # 3 separate arms
# Without Slurm (e.g. in the Docker image, see docker/README.md), run it with bash instead:
#   ARM=9,10,11 bash scripts/l40_pai_resume.sbatch
#
# ARM: a comma list of layers (9,10,11, the default; run name tag 9_10_11), one layer (7), an
# inclusive range (6-11), or control (= layer 11 with CONTROL=1). CONTROL=1 keeps the wrapping
# without ever adding dendrites.
# PAI schedule: PAI_PATIENCE (default 5) validations without improvement before the LR drops
# (--pai-patience), and at least PAI_N_EPOCHS_TO_SWITCH (default 15) validations in a neuron
# phase before PAI may switch to a dendrite phase (--pai-n-epochs-to-switch).
# W&B checkpoints (--wandb-checkpoints): the run keeps three checkpoints in the W&B model artifact
# checkpoint-<run name>: 'last' (uploaded at every validation), 'best' (best val acc@1) and
# 'pre_switch' (the last neuron-only state before PAI's first switch to dendrites, also kept on
# disk in <run>/archive/pre_switch/). Older versions are deleted. WANDB_CKPT=0 turns this off.
#
# RESUME: the pre-switch main.py checkpoint to continue from (--pai-resume). Its directory must
# also hold pai/best_model.pt. Default: the step-61000 snapshot of
# vit_tiny_pai_scratch_11_20pct_20261111_0433 (the start of the Delta/AWS resume runs) under
# <repo parent>/checkpoints/; copy that whole snapshot directory there from Delta, e.g.
#   rsync -a <delta>:/projects/bjto/shared/checkpoints/vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/1111_234822_step61000 \
#       <repo parent>/checkpoints/vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/
# RESUME_WANDB=<run name>: resume from that run's W&B pre-switch checkpoint instead, from any
# machine: downloads checkpoint-<run name>:pre_switch (from WANDB_ENTITY/WANDB_PROJECT in .env) to
# <repo parent>/checkpoints/<run name>/archive/pre_switch/ and sets RESUME to it.
# SCRATCH=1 trains from random initialization instead (no RESUME needed), with a linear LR warmup
# over PAI_WARMUP steps (default 2000).
#
# Same training settings as delta_pai_resume.sbatch: batch 1024 in one pass, no gradient
# accumulation, at LR 1e-3 (fits the L40's 48 GB in FP32 + TF32), validation every 250 steps.
# Other settings as there:
#   AMP=1, PAI_MODE=pb|gd, DENDRITE_LOSS=covariance, MAX_DENDRITES, PAI_TARGET,
#   CONTINUE_RUN=<run name> (continue a timed-out run; pass its original settings too).
# EPOCHS (default 350) is the total, including the epochs of the run RESUME comes from, and only a
# safety cap: PAI ends the run when it stops improving.
# Paths, if setup_machine.sh ran with non-default ones: REPO, VENV, DATA_VENV, DATA_DIR,
# PARQUET_DIR, LOG_ROOT (default <repo parent>/runs), LOCAL (the unpack directory).

set -euo pipefail
umask 002

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$REPO/main.py" ]; then
    echo "no main.py in $REPO: submit from the ViT-perforated repo root, or pass REPO=<repo>" >&2
    exit 1
fi
VENV=${VENV:-$REPO/.venv}
DATA_VENV=${DATA_VENV:-$REPO/.venv-data}
DATA_DIR=${DATA_DIR:-$(dirname "$REPO")}
PARQUET=${PARQUET_DIR:-$DATA_DIR/imagenet_1k_resized_256/data}
PYTHON=$VENV/bin/python
LOG_ROOT=${LOG_ROOT:-$DATA_DIR/runs}
# Outside Slurm: a timestamp as the job id, and all of this machine's cores.
JOB_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
LOCAL=${LOCAL:-${SLURM_TMPDIR:-/tmp/${USER:-$(id -un)}}/imagenet_$JOB_ID}

ARM=${ARM:-9,10,11}
PAI_TARGET=${PAI_TARGET:-ln_mlp}
EPOCHS=${EPOCHS:-350}
MAX_DENDRITES=${MAX_DENDRITES:-2}
PAI_PATIENCE=${PAI_PATIENCE:-5}
PAI_N_EPOCHS_TO_SWITCH=${PAI_N_EPOCHS_TO_SWITCH:-15}
CKPT_ARGS=()
[ "${WANDB_CKPT:-1}" = 1 ] && CKPT_ARGS=(--wandb-checkpoints)
if [ -n "${RESUME_WANDB:-}" ]; then
    RESUME=$DATA_DIR/checkpoints/$RESUME_WANDB/archive/pre_switch/checkpoint.pth.tar
fi
RESUME=${RESUME:-$DATA_DIR/checkpoints/vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/1111_234822_step61000/checkpoint.pth.tar}
# The source run's W&B id is its log directory name (<run>/archive/<snapshot>/ or <run>/checkpoints/).
SRC_DIR=${RESUME%%/archive/*}
[ "$SRC_DIR" = "$RESUME" ] && SRC_DIR=${RESUME%%/checkpoints/*}
SRC_RUN=$(basename "$SRC_DIR")

CONTROL=${CONTROL:-0}
ARM_TAG=$ARM
if [ "$ARM" = control ]; then
    LAYERS=(11)
    CONTROL=1
elif [[ $ARM =~ ^([0-9]+)-([0-9]+)$ ]]; then
    LAYERS=($(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"))
    [ ${#LAYERS[@]} -gt 0 ] || { echo "ARM=$ARM is an empty range" >&2; exit 1; }
elif [[ $ARM =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    IFS=, read -ra LAYERS <<< "$ARM"
    ARM_TAG=${ARM//,/_}
else
    echo "ARM must be a layer (11), a list (7,9,11), a range (6-11) or control" >&2; exit 1
fi
PAI_ARGS=(--pai-layers "${LAYERS[@]}")
if [ "$CONTROL" = 1 ]; then
    PAI_ARGS+=(--pai-control)
    [ "$ARM" = control ] || ARM_TAG=${ARM_TAG}_control
fi
PAI_MODE=${PAI_MODE:-pb}
MODE_ARGS=()
case "$PAI_MODE" in
    pb) ;;
    gd) MODE_ARGS=(--pai-gd) ;;
    *) echo "PAI_MODE must be pb or gd" >&2; exit 1 ;;
esac
LOSS_TAG=
case "${DENDRITE_LOSS:-default}" in
    default) ;;
    covariance) MODE_ARGS+=(--pai-dendrite-loss covariance); LOSS_TAG=_covloss ;;
    *) echo "DENDRITE_LOSS must be default or covariance" >&2; exit 1 ;;
esac
PRECISION=(--tf32)
PREC_TAG=fp32
if [ "${AMP:-0}" = 1 ]; then PRECISION=(--amp); PREC_TAG=amp; fi
SCRATCH=${SCRATCH:-0}
KIND=resume
[ "$SCRATCH" = 1 ] && KIND=scratch
RUN_PREFIX="vit_tiny_pai_${KIND}_${ARM_TAG}_${PAI_TARGET}_${PAI_MODE}${LOSS_TAG}_${PREC_TAG}_20pct_"
if [ -n "${CONTINUE_RUN:-}" ]; then
    case "$CONTINUE_RUN" in
        "$RUN_PREFIX"*) ;;
        *) echo "CONTINUE_RUN=$CONTINUE_RUN does not match these settings ($RUN_PREFIX*)" >&2; exit 1 ;;
    esac
    RUN_NAME=$CONTINUE_RUN
    CKPT_DIR=$LOG_ROOT/$RUN_NAME/checkpoints
    # The continued run overwrites checkpoint.pth.tar at its first validation; keep the start point.
    CONTINUE_FROM=$CKPT_DIR/continue_job${JOB_ID}.pth.tar
    cp "$CKPT_DIR/checkpoint.pth.tar" "$CONTINUE_FROM"
    STEP=$("$PYTHON" -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step'])" "$CONTINUE_FROM")
    WANDB_RUN=${RUN_NAME}_cont${JOB_ID}
    echo "Continuing $RUN_NAME from step $STEP as W&B run $WANDB_RUN"
    START_ARGS=(--resume "$CONTINUE_FROM" --wandb-name "$WANDB_RUN" --wandb-history-from "$RUN_NAME")
    # Own W&B dir: the original's wandb-resume.json (left by the timeout) would make
    # resume='auto' reach for the old run.
    WANDB_DIR=$LOG_ROOT/$RUN_NAME/$WANDB_RUN
else
    RUN_NAME="${RUN_PREFIX}$(date +%Y%m%d_%H%M)"
    if [ "$SCRATCH" = 1 ]; then
        START_ARGS=(--pai-warmup "${PAI_WARMUP:-2000}")
    else
        START_ARGS=(--pai-resume "$RESUME" --wandb-history-from "$SRC_RUN")
    fi
    WANDB_DIR=$LOG_ROOT/$RUN_NAME
fi

# Keys from the .env that setup_machine.sh checked (W&B token, PAI license, W&B entity/project).
set -a; source "$REPO/.env"; set +a
: "${PAIEMAIL:?PAIEMAIL is not set in $REPO/.env}"
: "${PAITOKEN:?PAITOKEN is not set in $REPO/.env}"
: "${WANDB_API_TOKEN:?WANDB_API_TOKEN is not set in $REPO/.env}"
test -x "$PYTHON" || { echo "no training venv at $VENV: run scripts/setup_machine.sh" >&2; exit 1; }
test -x "$DATA_VENV/bin/python" || { echo "no pyarrow venv at $DATA_VENV: run scripts/setup_machine.sh" >&2; exit 1; }
if ! compgen -G "$PARQUET/train-*.parquet" >/dev/null || ! compgen -G "$PARQUET/val-*.parquet" >/dev/null; then
    echo "no train-*/val-*.parquet shards in $PARQUET (set PARQUET_DIR)" >&2; exit 1
fi
if [ -n "${RESUME_WANDB:-}" ] && [ -z "${CONTINUE_RUN:-}" ] && [ "$SCRATCH" != 1 ]; then
    # Same layout as the run's archive/pre_switch/: checkpoint.pth.tar and pai/best_model.pt.
    "$PYTHON" - "$RESUME_WANDB" "$(dirname "$RESUME")" <<'PY'
import os, sys, wandb
run, dest = sys.argv[1:]
path = f"{os.environ.get('WANDB_ENTITY', 'IdlF26Group40')}/{os.environ.get('WANDB_PROJECT', 'ViT-perforated')}/checkpoint-{run}:pre_switch"
artifact = wandb.Api(api_key=os.environ['WANDB_API_TOKEN']).artifact(path, type='model')
artifact.download(dest)
print(f"=> downloaded {path} (step {artifact.metadata.get('step')}) to {dest}")
PY
fi
if [ -z "${CONTINUE_RUN:-}" ] && [ "$SCRATCH" != 1 ]; then
    if [ ! -r "$RESUME" ] || [ ! -r "$(dirname "$RESUME")/pai/best_model.pt" ]; then
        # PAI reloads pai/best_model.pt at the first switch.
        echo "RESUME=$RESUME or its pai/best_model.pt is missing: copy the snapshot directory" \
             "here (see the header), pass RESUME=<checkpoint>, or use SCRATCH=1" >&2
        exit 1
    fi
fi
mkdir -p "$LOG_ROOT"

# Unpack the parquet shards to node-local disk (faster than unpacking once on shared storage).
mkdir -p "$LOCAL"
avail_gb=$(df -BG --output=avail "$LOCAL" | tail -1 | tr -dc 0-9)
if [ "$avail_gb" -lt 40 ]; then
    echo "only ${avail_gb} GB free on $(df --output=target "$LOCAL" | tail -1); need ~30 GB (set LOCAL)" >&2
    exit 1
fi
trap 'rm -rf "$LOCAL"' EXIT
echo "Unpacking ImageNet from $PARQUET to $LOCAL ($avail_gb GB free)..."
start=$SECONDS
"$DATA_VENV/bin/python" "$REPO/scripts/parquet_to_imagefolder.py" "$PARQUET" "$LOCAL" --workers "$CPUS"
echo "Unpacked in $((SECONDS - start)) s"

export WANDB_DIR
export WANDB_ENTITY=${WANDB_ENTITY:-IdlF26Group40}
export WANDB_PROJECT=${WANDB_PROJECT:-ViT-perforated}
mkdir -p "$WANDB_DIR"
export OMP_NUM_THREADS=1

cd "$REPO"
echo "Run $RUN_NAME: layer(s) ${LAYERS[*]}, CONTROL=$CONTROL, patience $PAI_PATIENCE, n-epochs-to-switch $PAI_N_EPOCHS_TO_SWITCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -u main.py "$LOCAL" \
    --gpu 0 \
    --train-fraction 0.2 \
    --subset-seed 42 \
    --hidden-dim 192 \
    --num-layers 12 \
    --num-heads 3 \
    --patch-size 16 \
    --batch-size 1024 \
    --accum-freq 1 \
    "${PRECISION[@]}" \
    --mlp-head \
    --torchvision-inception-crop \
    --epochs "$EPOCHS" \
    --lr 1e-3 \
    --log-steps 250 \
    --workers $((CPUS > 4 ? CPUS - 2 : 2)) \
    "${START_ARGS[@]}" \
    "${PAI_ARGS[@]}" \
    ${MODE_ARGS[@]+"${MODE_ARGS[@]}"} \
    --pai-target "$PAI_TARGET" \
    --max-dendrites "$MAX_DENDRITES" \
    --pai-patience "$PAI_PATIENCE" \
    --pai-n-epochs-to-switch "$PAI_N_EPOCHS_TO_SWITCH" \
    --logs "$LOG_ROOT" \
    --name "$RUN_NAME" \
    --report-to wandb \
    ${CKPT_ARGS[@]+"${CKPT_ARGS[@]}"}
