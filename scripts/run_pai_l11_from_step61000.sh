#!/bin/bash
# One ViT-Tiny PAI run: dendrites on encoder block 11 (ln_mlp, Perforated Backpropagation with the
# default ImprovedCovarianceDendriteLoss), continuing the neuron phase of the step-61000 pre-switch
# snapshot of vit_tiny_pai_scratch_11_20pct_20261111_0433. PAI decides the switches and ends the run.
# Settings of the Nov 13-17 Delta plateau runs: batch 512, LR 5e-4, validation every 500 steps (one
# epoch of the 20% subset), patience 2, at least 10 validations before a switch, up to 2 dendrites,
# FP32 + TF32, at most 300 epochs in total.
#
#   GPU=0 ./scripts/run_pai_l11_from_step61000.sh
#   CONTINUE_RUN=<run name> GPU=0 ./scripts/run_pai_l11_from_step61000.sh   # after a crash
#
# RESUME: the snapshot's checkpoint.pth.tar (its directory must also hold pai/best_model.pt).
# Default: <repo parent>/checkpoints/vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/1111_234822_step61000/.
# LOGS: run directories (default ~/runs). WANDB=0: no W&B.
# DATA, GPU, WORKERS, WANDB_ENTITY / WANDB_PROJECT: see scripts/machine_env.sh
# (WANDB_PROJECT defaults to ViT-perforated-plateau here, where the earlier plateau runs are).
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
export WANDB_PROJECT=${WANDB_PROJECT:-ViT-perforated-plateau}
# Sets DATA, WORKERS, PYTHON, the PAI credentials, the W&B entity and the visible GPU.
source "$REPO/scripts/machine_env.sh"
: "${PAIEMAIL:?PAIEMAIL is not set}"
: "${PAITOKEN:?PAITOKEN is not set}"
LOGS=${LOGS:-$HOME/runs}
SRC_RUN=vit_tiny_pai_scratch_11_20pct_20261111_0433
RESUME=${RESUME:-$(dirname "$REPO")/checkpoints/$SRC_RUN/archive/1111_234822_step61000/checkpoint.pth.tar}
PREFIX=vit_tiny_pai_resume_11_ln_mlp_pb_fp32_b512_lr5e-4_20pct_

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
    if [ ! -r "$RESUME" ] || [ ! -r "$(dirname "$RESUME")/pai/best_model.pt" ]; then
        echo "RESUME=$RESUME or its pai/best_model.pt is missing" >&2; exit 1
    fi
    RUN_NAME=${PREFIX}$(date +%Y%m%d_%H%M)
    # The source run's history is in another W&B project.
    START_ARGS=(--pai-resume "$RESUME" --wandb-history-from "ViT-perforated-amp/$SRC_RUN")
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
    --epochs 300 --lr 5e-4 --log-steps 500 --workers "$WORKERS" \
    "${START_ARGS[@]}" \
    --pai-layers 11 --pai-target ln_mlp --max-dendrites 2 \
    --pai-patience 2 --pai-n-epochs-to-switch 10 \
    --logs "$LOGS" --name "$RUN_NAME" ${REPORT[@]+"${REPORT[@]}"} 2>&1 | tee -a "$WANDB_DIR/train.log"
