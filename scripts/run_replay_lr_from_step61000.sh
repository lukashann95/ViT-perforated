#!/bin/bash
# Control for run_pai_l11_from_step61000.sh: the same ViT-Tiny without PAI or dendrites, trained on
# the LR schedule that PAI run followed (vit_tiny_pai_resume_11_ln_mlp_pb_fp32_b512_lr5e-4_20pct_
# 20261126_0316), with its warm restarts: at each restart the best model of the cycle is reloaded
# and AdamW starts fresh, as PAI does. Shows how much of that run's gain the restarts alone give.
# The schedule is scripts/replay_lr/<PAI run>.json (see lr_replay.py and scripts/build_replay_lr.py):
# PAI's neuron phases on the path to its final model, without its dendrite phases (neurons frozen)
# and abandoned LR-search trials, so 50500 steps instead of 83000.
# Starts from the step-61000 snapshot's pai/best_model.pt (35.90%), the model PAI loaded at its
# first switch. Batch 512, --lr 5e-4 (only sets the weight decay: --wd 1e-4 / 5e-4 = 0.2, as in
# the PAI run), validation every 500 steps, FP32 + TF32.
#
#   GPU=0 ./scripts/run_replay_lr_from_step61000.sh
#   CONTINUE_RUN=<run name> GPU=0 ./scripts/run_replay_lr_from_step61000.sh   # after a crash
#
# INIT: the snapshot's pai/best_model.pt. Default: <repo parent>/checkpoints/
# vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/1111_234822_step61000/pai/best_model.pt.
# LOGS: run directories (default ~/runs). WANDB=0: no W&B.
# DATA, GPU, WORKERS, WANDB_ENTITY / WANDB_PROJECT: see scripts/machine_env.sh
# (WANDB_PROJECT defaults to ViT-perforated-plateau here, next to the PAI run).
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
export WANDB_PROJECT=${WANDB_PROJECT:-ViT-perforated-plateau}
# Sets DATA, WORKERS, PYTHON, the W&B entity and the visible GPU.
source "$REPO/scripts/machine_env.sh"
LOGS=${LOGS:-$HOME/runs}
SRC_RUN=vit_tiny_pai_scratch_11_20pct_20261111_0433
PAI_RUN=vit_tiny_pai_resume_11_ln_mlp_pb_fp32_b512_lr5e-4_20pct_20261126_0316
SCHEDULE=$REPO/scripts/replay_lr/$PAI_RUN.json
INIT=${INIT:-$(dirname "$REPO")/checkpoints/$SRC_RUN/archive/1111_234822_step61000/pai/best_model.pt}
PREFIX=vit_tiny_replay_lr_of_pai_resume_11_b512_lr5e-4_20pct_

if [ -n "${CONTINUE_RUN:-}" ]; then
    case "$CONTINUE_RUN" in "$PREFIX"*) ;; *) echo "CONTINUE_RUN must start with $PREFIX" >&2; exit 1 ;; esac
    RUN_NAME=$CONTINUE_RUN
    # main.py overwrites checkpoint.pth.tar at its first validation, so keep the start point.
    CONTINUE_FROM=$LOGS/$RUN_NAME/checkpoints/continue_$(date +%Y%m%d_%H%M%S).pth.tar
    cp "$LOGS/$RUN_NAME/checkpoints/checkpoint.pth.tar" "$CONTINUE_FROM"
    WANDB_RUN=${RUN_NAME}_cont$(date +%m%d%H%M)
    START_ARGS=(--resume "$CONTINUE_FROM" --wandb-name "$WANDB_RUN" --wandb-history-from "$RUN_NAME")
    # Own W&B directory, so W&B does not pick up the crashed run's resume file.
    export WANDB_DIR=$LOGS/$RUN_NAME/$WANDB_RUN
else
    test -r "$INIT" || { echo "cannot read INIT=$INIT" >&2; exit 1; }
    RUN_NAME=${PREFIX}$(date +%Y%m%d_%H%M)
    # The schedule starts at step 61000; copy the source run's history up to there, as the PAI run did.
    START_ARGS=(--init "$INIT" --wandb-history-from "ViT-perforated-amp/$SRC_RUN")
    export WANDB_DIR=$LOGS/$RUN_NAME
fi
REPORT=()
if [ "${WANDB:-1}" = 1 ]; then REPORT=(--report-to wandb --wandb-checkpoints); fi
mkdir -p "$WANDB_DIR"

cd "$REPO"
echo "=== $RUN_NAME on GPU $CUDA_VISIBLE_DEVICES, data $DATA, $WORKERS workers, commit $(git log --oneline -1 2>/dev/null || echo unknown)"
"$PYTHON" -u main.py "$DATA" \
    --gpu 0 --train-fraction 0.2 --subset-seed 42 \
    --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16 \
    --batch-size 512 --accum-freq 1 --tf32 --mlp-head --torchvision-inception-crop \
    --lr 5e-4 --log-steps 500 --workers "$WORKERS" \
    --replay-lr "$SCHEDULE" "${START_ARGS[@]}" \
    --logs "$LOGS" --name "$RUN_NAME" ${REPORT[@]+"${REPORT[@]}"} 2>&1 | tee -a "$WANDB_DIR/train.log"
