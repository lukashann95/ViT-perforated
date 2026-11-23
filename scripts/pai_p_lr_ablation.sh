#!/bin/bash
#SBATCH --job-name=vit-plr-ablation
#SBATCH --account=bjto-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/projects/bjto/lhann/runs/slurm-%j.out
# Dendrite-phase LR ablation for one plain (non-PAI) ViT-Tiny checkpoint: for each LR scale K,
# perforate block 11 (TARGET, default its pre-norm and whole MLP) and train only the first
# Perforated Backpropagation dendrite (p) phase for 25 epochs, cosine from K x PEAK_LR down to 0,
# with no weight decay. Look at PAI's PB-score plots (<run>/pai_images, W&B), not validation
# accuracy. The runs go one after another:
#   CKPT=/path/model_best.pth.tar PEAK_LR=1e-3 BATCH=1024 scripts/pai_p_lr_ablation.sh
# As one task-spooler job on riddle GPU 1:
#   forgeq-gpu 1 -L plr env GPU=1 CKPT=... PEAK_LR=2.5e-4 BATCH=256 ~/ViT-perforated/scripts/pai_p_lr_ablation.sh
# On Delta, from the repo root (the #SBATCH lines are comments elsewhere):
#   sbatch --export=ALL,CKPT=...,PEAK_LR=1e-3,BATCH=1024 scripts/pai_p_lr_ablation.sh
#
# PEAK_LR is the neuron run's peak LR (the start of its cosine), for its effective batch BATCH.
# With ACCUM=k the p phase steps once per BATCH/k microbatch, and --pai-p-lr-scaling linear
# trains the candidates at K x PEAK_LR / k. The four setups:
#   1024 in one pass:            PEAK_LR=1e-3   BATCH=1024
#   1024 neurons, 256 dendrites: PEAK_LR=1e-3   BATCH=1024 ACCUM=4  (candidates at K x 2.5e-4, as
#                                                                    the linear-scaled 256 setup)
#   256, linear LR scaling:      PEAK_LR=2.5e-4 BATCH=256
#   256, sqrt LR scaling:        PEAK_LR=5e-4   BATCH=256
# Env: CKPT (or CKPT_WANDB), PEAK_LR, BATCH (required); ACCUM (default 1); SCALES (default "0.01 0.1 1 10 100");
# GD=1 uses PAI's gradient-descent dendrites (--pai-gd) instead of Perforated Backpropagation;
#   run names then carry "gd" instead of "pb". GD has no separate dendrite (p) phase: after the
#   forced switch the dendrite is trained together with the (unfrozen) neurons in a neuron phase,
#   so K x PEAK_LR and WD apply to the whole network, there are no PB scores (judge by val/acc@1),
#   --pai-p-lr-scaling has no effect, and --accum-freq accumulates exactly (ACCUM=4 at BATCH=1024
#   is one true 1024 batch, which is what makes it fit in 24 GB);
# MAX_DENDRITES (default 1; >1 needs GD=1): dendrite rounds. The forced switch at the start adds
#   the first; each later round is added by a fixed switch after another P_EPOCHS epochs. Every
#   round gets its own P_EPOCHS-epoch cosine from K x PEAK_LR to 0 (PAI restarts the scheduler at
#   each switch), so a run lasts MAX_DENDRITES x P_EPOCHS epochs;
# WD (default 0): weight decay. With PB the neurons are frozen, so 0 only affects candidates;
#   with GD it applies to every weight matrix (the baseline used 1e-4);
# TRAIN_FRACTION (default 0.2, the stratified subset with seed 42; 1 = all of ImageNet; run names
#   carry _f<fraction>). P_EPOCHS counts epochs of that training set: 5 full-data epochs are as
#   many steps as 25 subset epochs;
# P_EPOCHS (default 25); LAYER (default 11); TARGET (--pai-target: ln_mlp, ln_fc1, fc1, fc2, mlp, qk, qkv;
# default ln_mlp); LOGS (default ~/runs; Delta: /projects/bjto/$USER/runs); WANDB=0 to skip W&B.
# DATA, GPU, WORKERS, WANDB_ENTITY / WANDB_PROJECT: see scripts/machine_env.sh
# (defaults ~/data/imagenet, 0, CPU cores - 1, IdlF26Group40 / ViT-perforated; Delta unpacks the
# dataset to node-local disk once for all scales).
# CKPT_WANDB=<run name> instead of CKPT downloads that run's W&B checkpoint artifact
# (checkpoint-<run name>:last, from runs with --wandb-checkpoints) to $LOGS/ckpts/<run name>/.
# Needs PAIEMAIL/PAITOKEN (environment, the repo's .env or ~/.pai_env); W&B reads
# WANDB_API_TOKEN from .env.
set -euo pipefail
[ -n "${CKPT:-}${CKPT_WANDB:-}" ] || { echo "set CKPT to a plain main.py checkpoint, or CKPT_WANDB to its run name" >&2; exit 1; }
: "${PEAK_LR:?set PEAK_LR to the peak LR of the neuron run}"
: "${BATCH:?set BATCH to 1024 or 256}"
ACCUM=${ACCUM:-1}
SCALES=${SCALES:-0.01 0.1 1 10 100}
P_EPOCHS=${P_EPOCHS:-25}
LAYER=${LAYER:-11}
TARGET=${TARGET:-ln_mlp}
TRAIN_FRACTION=${TRAIN_FRACTION:-0.2}
# Training images per epoch, as main.py reports them (the subset rounds down per class).
case "$TRAIN_FRACTION" in
    0.2) TRAIN_IMAGES=256190 ;;
    1)   TRAIN_IMAGES=1281167 ;;
    *)   echo "TRAIN_FRACTION must be 0.2 or 1 (the training-set size of other fractions is not known here)" >&2; exit 1 ;;
esac
WD=${WD:-0}
GD=${GD:-0}
MAX_DENDRITES=${MAX_DENDRITES:-1}
if [ "$GD" = 1 ]; then GD_ARGS=(--pai-gd); MODE=gd; else GD_ARGS=(); MODE=pb; fi
if [ "$MAX_DENDRITES" -lt 1 ]; then echo "MAX_DENDRITES must be >= 1" >&2; exit 1; fi
if [ "$MAX_DENDRITES" -gt 1 ] && [ "$GD" != 1 ]; then
    # PB alternates p and n phases, so its schedule for several rounds is not this one.
    echo "MAX_DENDRITES > 1 is only set up for GD=1" >&2; exit 1
fi
TOTAL_EPOCHS=$((P_EPOCHS * MAX_DENDRITES))
if [ "$MAX_DENDRITES" -gt 1 ]; then
    # GD stays in neuron phases; the next round comes after P_EPOCHS validations.
    SWITCH_ARGS=(--pai-fixed-switch "$P_EPOCHS" --pai-p-epochs-to-switch "$P_EPOCHS")
else
    # With --pai-fixed-switch the p phase lasts exactly --pai-p-epochs-to-switch validations.
    SWITCH_ARGS=(--pai-fixed-switch 1000 --pai-p-epochs-to-switch "$P_EPOCHS")
fi

# sbatch runs a copy of this script, so under Slurm the repo is where sbatch was called from.
REPO=${REPO:-$(cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" && pwd)}
test -f "$REPO/scripts/machine_env.sh" || { echo "No $REPO/scripts/machine_env.sh (with sbatch, submit from the repo root)" >&2; exit 1; }
# Sets DATA, WORKERS, PYTHON, the PAI credentials, the W&B entity/project and the visible GPU.
source "$REPO/scripts/machine_env.sh"
LOGS=${LOGS:-$HOME/runs}
: "${PAIEMAIL:?PAIEMAIL is not set}"
: "${PAITOKEN:?PAITOKEN is not set}"
cd "$REPO"
if [ -z "${CKPT:-}" ]; then
    CKPT=$LOGS/ckpts/$CKPT_WANDB/checkpoint.pth.tar
    if [ ! -r "$CKPT" ]; then
        # Download next to it and rename, so parallel jobs never read a half-written file.
        mkdir -p "$(dirname "$CKPT")"
        TMP_DL=$(mktemp -d "$(dirname "$CKPT")/.download.XXXXXX")
        "$PYTHON" - "$CKPT_WANDB" "$TMP_DL" <<'PY'
import os, sys, wandb
run, dest = sys.argv[1:]
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/checkpoint-{run}:last"
artifact = wandb.Api(api_key=os.environ.get('WANDB_API_TOKEN') or None).artifact(path, type='model')
artifact.download(dest)
print(f"=> downloaded {path} (step {artifact.metadata.get('step')}, "
      f"acc@1 {artifact.metadata.get('acc1')}) to {dest}")
PY
        mv -f "$TMP_DL/checkpoint.pth.tar" "$CKPT"
        rm -rf "$TMP_DL"
    fi
fi
test -r "$CKPT" || { echo "cannot read CKPT=$CKPT" >&2; exit 1; }
COMMIT=$(git log --oneline -1 2>/dev/null || echo "unknown commit (no .git)")
REPORT=()
if [ "${WANDB:-1}" = 1 ]; then REPORT=(--report-to wandb); fi

# One validation per epoch of the training set, so PAI's epochs are real epochs;
# stop right after the P_EPOCHS-th validation, where the fixed switch ends the p phase.
LOG_STEPS=$((TRAIN_IMAGES / BATCH))
MAX_STEPS=$((TOTAL_EPOCHS * LOG_STEPS))
STAMP=$(date +%Y%m%d_%H%M)
FAILED=()
for K in $SCALES; do
    LR=$(awk -v a="$PEAK_LR" -v k="$K" 'BEGIN { printf "%.6g", a * k }')
    RUN_NAME=vit_tiny_pai_plr_${MODE}_l${LAYER}_${TARGET}_b${BATCH}_acc${ACCUM}_lr${PEAK_LR}_wd${WD}_d${MAX_DENDRITES}_f${TRAIN_FRACTION}_k${K}_${STAMP}
    export WANDB_DIR=$LOGS/$RUN_NAME
    mkdir -p "$WANDB_DIR"
    echo "=== $RUN_NAME ($MODE): LR $LR ($K x $PEAK_LR, divided by ACCUM=$ACCUM in the p phase) -> 0" \
         "over $P_EPOCHS epochs per dendrite round, $MAX_DENDRITES round(s), $TOTAL_EPOCHS epochs, from $CKPT, commit $COMMIT"
    # --pai-init perforates the loaded model and forces the switch into the p phase at step 0.
    # Neurons are frozen in the p phase, so --wd 0 means no weight decay on the candidates.
    if ! "$PYTHON" -u main.py "$DATA" \
        --gpu 0 --train-fraction "$TRAIN_FRACTION" --subset-seed 42 \
        --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16 \
        --batch-size "$BATCH" --accum-freq "$ACCUM" --tf32 --mlp-head --torchvision-inception-crop \
        --epochs "$TOTAL_EPOCHS" --max-steps "$MAX_STEPS" --log-steps "$LOG_STEPS" \
        --lr "$LR" --wd "$WD" --workers "$WORKERS" \
        --pai-init "$CKPT" --pai-layers "$LAYER" --pai-target "$TARGET" --max-dendrites "$MAX_DENDRITES" "${GD_ARGS[@]}" \
        --pai-schedule cosine --pai-cosine-epochs "$P_EPOCHS" --pai-p-lr-scaling linear \
        "${SWITCH_ARGS[@]}" \
        --wandb-notes "p-phase LR ablation ($MODE) from $CKPT, K=$K, peak LR $PEAK_LR, wd $WD, accum $ACCUM, train fraction $TRAIN_FRACTION, dendrites $MAX_DENDRITES, target $TARGET" \
        --logs "$LOGS" --name "$RUN_NAME" "${REPORT[@]}" 2>&1 | tee "$WANDB_DIR/train.log"; then
        FAILED+=("$K")   # e.g. K=100 diverging; keep going with the other scales
    fi
done
if [ ${#FAILED[@]} -gt 0 ]; then echo "Failed scales: ${FAILED[*]}" >&2; exit 1; fi
