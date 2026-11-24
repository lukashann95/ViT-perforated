#!/bin/bash
# Neuron (n) phase after a dendrite-phase ablation: continue a pai_p_lr_ablation.sh run from the
# checkpoint its final validation saved (dendrite installed, neuron phase) and train the whole
# network for N_EPOCHS epochs with normal augmentation, cosine from LR to 0, weight decay WD.
# The saved optimizer and schedule are replaced (--pai-new-schedule); judge by val/acc@1.
# As one task-spooler job on riddle GPU 1:
#   forgeq-gpu 1 env GPU=1 SRC=<p-phase run name> LR=1e-4 ~/ViT-perforated/scripts/pai_n_phase.sh
# Env: SRC (required): the p-phase run, read from $LOGS/$SRC/checkpoints/checkpoint.pth.tar;
# LR (required); N_EPOCHS (default 10, epochs of the training set); WD (default 1e-4, the baseline's
# at batch 1024); BATCH (1024), ACCUM (4), LAYER (11), TARGET (ln_mlp), TRAIN_FRACTION (1) must
# match the p-phase run. UPLOAD_CKPTS=1 uploads checkpoints (--wandb-checkpoints); WANDB=0 skips W&B.
# LOGS, DATA, GPU, WORKERS: as in pai_p_lr_ablation.sh.
set -euo pipefail
: "${SRC:?set SRC to the p-phase run name}"
: "${LR:?set LR, the n-phase starting LR}"
N_EPOCHS=${N_EPOCHS:-10}
WD=${WD:-1e-4}
BATCH=${BATCH:-1024}
ACCUM=${ACCUM:-4}
LAYER=${LAYER:-11}
TARGET=${TARGET:-ln_mlp}
TRAIN_FRACTION=${TRAIN_FRACTION:-1}
case "$TRAIN_FRACTION" in
    0.2) TRAIN_IMAGES=256190 ;;
    1)   TRAIN_IMAGES=1281167 ;;
    *)   echo "TRAIN_FRACTION must be 0.2 or 1" >&2; exit 1 ;;
esac

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
source "$REPO/scripts/machine_env.sh"
LOGS=${LOGS:-$HOME/runs}
: "${PAIEMAIL:?PAIEMAIL is not set}"
: "${PAITOKEN:?PAITOKEN is not set}"
cd "$REPO"
CKPT=$LOGS/$SRC/checkpoints/checkpoint.pth.tar
test -r "$CKPT" || { echo "cannot read $CKPT" >&2; exit 1; }
COMMIT=$(git log --oneline -1 2>/dev/null || echo "unknown commit")
REPORT=()
if [ "${WANDB:-1}" = 1 ]; then REPORT=(--report-to wandb); fi
if [ "${UPLOAD_CKPTS:-0}" = 1 ]; then REPORT+=(--wandb-checkpoints); fi

# One validation per epoch; the cosine steps once per validation. The run continues the source's
# step count, so --epochs covers the source's epochs plus N_EPOCHS.
LOG_STEPS=$((TRAIN_IMAGES / BATCH))
START_STEP=$("$PYTHON" -c "import torch, sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step'])" "$CKPT")
START_EPOCHS=$(( (START_STEP + LOG_STEPS / 2) / LOG_STEPS ))
RUN_NAME=${SRC}_n${N_EPOCHS}_lr${LR}_wd${WD}
export WANDB_DIR=$LOGS/$RUN_NAME
mkdir -p "$WANDB_DIR"
echo "=== $RUN_NAME: n phase from $CKPT (step $START_STEP), LR $LR -> 0 over $N_EPOCHS epochs, wd $WD, commit $COMMIT"
# --pai-fixed-switch 1000: no further switch (max dendrites is reached anyway).
"$PYTHON" -u main.py "$DATA" \
    --gpu 0 --train-fraction "$TRAIN_FRACTION" --subset-seed 42 \
    --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16 \
    --batch-size "$BATCH" --accum-freq "$ACCUM" --tf32 --mlp-head --torchvision-inception-crop \
    --epochs $((START_EPOCHS + N_EPOCHS)) --max-steps $((N_EPOCHS * LOG_STEPS)) --log-steps "$LOG_STEPS" \
    --lr "$LR" --wd "$WD" --workers "$WORKERS" \
    --resume "$CKPT" --pai-new-schedule --pai-layers "$LAYER" --pai-target "$TARGET" --max-dendrites 1 \
    --pai-schedule cosine --pai-cosine-epochs "$N_EPOCHS" --pai-p-lr-scaling linear --pai-fixed-switch 1000 \
    --wandb-notes "n phase after $SRC: $N_EPOCHS epochs, LR $LR -> 0, wd $WD, augmented, $COMMIT" \
    --logs "$LOGS" --name "$RUN_NAME" "${REPORT[@]}" 2>&1 | tee "$WANDB_DIR/train.log"
