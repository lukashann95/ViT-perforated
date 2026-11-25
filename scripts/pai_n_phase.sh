#!/bin/bash
# Neuron (n) phase after a dendrite-phase ablation: continue a pai_p_lr_ablation.sh run from the
# checkpoint its final validation saved (dendrite installed, neuron phase) and train the whole
# network for N_EPOCHS epochs with normal augmentation, cosine from LR to 0, weight decay WD.
# The saved optimizer and schedule are replaced (--pai-new-schedule); judge by val/acc@1.
# CONTROL=1 is the no-dendrite control: CKPT=<plain main.py checkpoint> (the one the p-phase run
# perforated, so the same neurons) with the same n phase under --pai-control, from step 0.
# CKPT_WANDB=<run name> instead of CKPT uses $LOGS/ckpts/<run name>/checkpoint.pth.tar, downloading
# that run's W&B artifact checkpoint-<run name>:last there if missing.
# As one task-spooler job on riddle GPU 1:
#   forgeq-gpu 1 env GPU=1 SRC=<p-phase run name> LR=1e-4 ~/ViT-perforated/scripts/pai_n_phase.sh
# Env: SRC (required unless CONTROL=1): the p-phase run, read from $LOGS/$SRC/checkpoints/checkpoint.pth.tar
# (downloaded from W&B artifact checkpoint-$SRC:last if missing: p-phase runs with UPLOAD_CKPTS=1);
# LR (required); N_EPOCHS (default 10, epochs of the training set); WD (default 1e-4, the baseline's
# at batch 1024); BATCH (1024), ACCUM (4), LAYER (11; several blocks space-separated), TARGET (ln_mlp), TRAIN_FRACTION (1) must
# match the p-phase run. UPLOAD_CKPTS=1 uploads checkpoints (--wandb-checkpoints); WANDB=0 skips W&B.
# CLEAN=1 trains the n phase without augmentation, as pai_p_lr_ablation.sh's CLEAN=1 does for dendrite phases
# (no MixUp, no RandAugment, the crop takes the whole image; flip and aspect ratio remain); names carry _n<N>_clean.
# RUN_SUFFIX (e.g. _rep2) is appended to the run name: needed to repeat a dendrite n phase from the same SRC,
# whose name is otherwise identical (same log folder and W&B run).
# LOGS, DATA, GPU, WORKERS: as in pai_p_lr_ablation.sh.
set -euo pipefail
CONTROL=${CONTROL:-0}
if [ "$CONTROL" = 1 ]; then
    [ -n "${CKPT:-}${CKPT_WANDB:-}" ] || { echo "set CKPT to the plain checkpoint the p-phase run started from, or CKPT_WANDB to its run name" >&2; exit 1; }
else
    : "${SRC:?set SRC to the p-phase run name}"
fi
: "${LR:?set LR, the n-phase starting LR}"
N_EPOCHS=${N_EPOCHS:-10}
WD=${WD:-1e-4}
BATCH=${BATCH:-1024}
ACCUM=${ACCUM:-4}
LAYER=${LAYER:-11}
read -ra LAYERS <<< "$LAYER"
LAYER_TAG=$(IFS=-; echo "${LAYERS[*]}")
TARGET=${TARGET:-ln_mlp}
TRAIN_FRACTION=${TRAIN_FRACTION:-1}
CLEAN=${CLEAN:-0}
if [ "$CLEAN" = 1 ]; then
    CLEAN_ARGS=(--mixup-alpha 0 --no-randaug --lower-scale 1 --upper-scale 1); CLEAN_TAG=_clean
else
    CLEAN_ARGS=(); CLEAN_TAG=
fi
case "$TRAIN_FRACTION" in
    0.2) TRAIN_IMAGES=256190 ;;
    1)   TRAIN_IMAGES=1281167 ;;
    *)   echo "TRAIN_FRACTION must be 0.2 or 1" >&2; exit 1 ;;
esac

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
source "$REPO/scripts/machine_env.sh"
source "$REPO/scripts/wandb_ckpt.sh"
LOGS=${LOGS:-$HOME/runs}
: "${PAIEMAIL:?PAIEMAIL is not set}"
: "${PAITOKEN:?PAITOKEN is not set}"
cd "$REPO"
if [ "$CONTROL" != 1 ]; then
    CKPT=$LOGS/$SRC/checkpoints/checkpoint.pth.tar
    fetch_wandb_ckpt "$SRC" "$CKPT"
elif [ -z "${CKPT:-}" ]; then
    CKPT=$LOGS/ckpts/$CKPT_WANDB/checkpoint.pth.tar
    fetch_wandb_ckpt "$CKPT_WANDB" "$CKPT"
fi
test -r "$CKPT" || { echo "cannot read $CKPT" >&2; exit 1; }
COMMIT=$(git log --oneline -1 2>/dev/null || echo "unknown commit")
REPORT=()
if [ "${WANDB:-1}" = 1 ]; then REPORT=(--report-to wandb); fi
if [ "${UPLOAD_CKPTS:-0}" = 1 ]; then REPORT+=(--wandb-checkpoints); fi

# One validation per epoch; the cosine steps once per validation. The run continues the source's
# step count, so --epochs covers the source's epochs plus N_EPOCHS.
LOG_STEPS=$((TRAIN_IMAGES / BATCH))
if [ "$CONTROL" = 1 ]; then
    # --pai-init starts at step 0 and validates the loaded model first.
    START_STEP=0
    INIT_ARGS=(--pai-init "$CKPT" --pai-control)
    RUN_NAME=vit_tiny_pai_nctrl_l${LAYER_TAG}_${TARGET}_b${BATCH}_acc${ACCUM}_f${TRAIN_FRACTION}_n${N_EPOCHS}${CLEAN_TAG}_lr${LR}_wd${WD}_$(date +%Y%m%d_%H%M)${RUN_SUFFIX:-}
    SRC="control from $CKPT"
else
    START_STEP=$("$PYTHON" -c "import torch, sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['step'])" "$CKPT")
    INIT_ARGS=(--resume "$CKPT" --pai-new-schedule)
    RUN_NAME=${SRC}_n${N_EPOCHS}${CLEAN_TAG}_lr${LR}_wd${WD}${RUN_SUFFIX:-}
fi
# W&B rejects run IDs (= our run names) over 128 characters: compress an all-blocks list, else stop.
if [ ${#RUN_NAME} -gt 128 ]; then RUN_NAME=${RUN_NAME//l0-1-2-3-4-5-6-7-8-9-10-11_/l0to11_}; fi
if [ ${#RUN_NAME} -gt 128 ]; then echo "run name is ${#RUN_NAME} characters, W&B allows 128: $RUN_NAME" >&2; exit 1; fi
START_EPOCHS=$(( (START_STEP + LOG_STEPS / 2) / LOG_STEPS ))
export WANDB_DIR=$LOGS/$RUN_NAME
mkdir -p "$WANDB_DIR"
echo "=== $RUN_NAME: n phase from $CKPT (step $START_STEP), LR $LR -> 0 over $N_EPOCHS epochs, wd $WD, commit $COMMIT"
# --pai-fixed-switch 1000: no further switch (max dendrites is reached anyway).
"$PYTHON" -u main.py "$DATA" \
    --gpu 0 --train-fraction "$TRAIN_FRACTION" --subset-seed 42 \
    --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16 \
    --batch-size "$BATCH" --accum-freq "$ACCUM" --tf32 --mlp-head --torchvision-inception-crop \
    --epochs $((START_EPOCHS + N_EPOCHS)) --max-steps $((N_EPOCHS * LOG_STEPS)) --log-steps "$LOG_STEPS" \
    --lr "$LR" --wd "$WD" --workers "$WORKERS" "${CLEAN_ARGS[@]}" \
    "${INIT_ARGS[@]}" --pai-layers "${LAYERS[@]}" --pai-target "$TARGET" --max-dendrites 1 \
    --pai-schedule cosine --pai-cosine-epochs "$N_EPOCHS" --pai-p-lr-scaling linear --pai-fixed-switch 1000 \
    --wandb-notes "n phase after $SRC: $N_EPOCHS epochs, LR $LR -> 0, wd $WD, $([ "$CLEAN" = 1 ] && echo clean || echo augmented), $COMMIT" \
    --logs "$LOGS" --name "$RUN_NAME" "${REPORT[@]}" 2>&1 | tee "$WANDB_DIR/train.log"
