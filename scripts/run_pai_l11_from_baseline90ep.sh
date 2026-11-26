#!/bin/bash
# One ViT-Tiny PAI run: dendrites on encoder block 11 (ln_mlp, Perforated Backpropagation with the
# default ImprovedCovarianceDendriteLoss), starting from the final checkpoint of the plain 90-epoch
# cosine baseline vit_tiny_tf32_1gpu_20pct_90ep_subset42_riddle_20261121_040828 (42.96% val acc@1).
# --pai-init loads only the weights (the cosine optimizer/scheduler state is dropped), perforates the
# model and switches into the first dendrite phase at the first validation (step 0), since the
# model has already converged. After that PAI's plateau schedule decides the switches and ends the run.
#
# Same batch, LR and weight decay as the baseline: batch 1024 as 4 x 256 microbatches (neuron phases
# accumulate them into one step; dendrite phases step after every 256-image microbatch), LR 1e-3,
# --wd 1e-4 (main divides it by the LR, so 0.1 in the optimizer, as logged by the baseline). PAI schedule as in run_pai_l11_from_step61000.sh: validation every epoch
# (250 steps), patience 2, at least 10 validations before a switch, up to 2 dendrites, FP32 + TF32.
#
#   GPU=1 ./scripts/run_pai_l11_from_baseline90ep.sh
#   CONTINUE_RUN=<run name> GPU=1 ./scripts/run_pai_l11_from_baseline90ep.sh   # after a crash
#
# The checkpoint is downloaded once from W&B (IdlF26Group40/ViT-perforated, artifact
# checkpoint-<baseline>:last) to $LOGS/ckpts/<baseline>/; CKPT=<path> uses a local copy instead.
# LOGS: run directories (default ~/runs). WANDB=0: no W&B.
# DATA, GPU, WORKERS, WANDB_ENTITY / WANDB_PROJECT: see scripts/machine_env.sh
# (WANDB_PROJECT defaults to ViT-perforated-plateau here).
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
export WANDB_PROJECT=${WANDB_PROJECT:-ViT-perforated-plateau}
# Sets DATA, WORKERS, PYTHON, the PAI credentials, the W&B entity and the visible GPU.
source "$REPO/scripts/machine_env.sh"
: "${PAIEMAIL:?PAIEMAIL is not set}"
: "${PAITOKEN:?PAITOKEN is not set}"
LOGS=${LOGS:-$HOME/runs}
BASELINE=vit_tiny_tf32_1gpu_20pct_90ep_subset42_riddle_20261121_040828
PREFIX=vit_tiny_pai_init_l11_ln_mlp_pb_fp32_b1024_acc4_lr1e-3_from_base90ep_

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
    if [ -z "${CKPT:-}" ]; then
        CKPT=$LOGS/ckpts/$BASELINE/checkpoint.pth.tar
        if [ ! -r "$CKPT" ]; then
            # Download next to it and rename, so a crash never leaves a half-written file.
            mkdir -p "$(dirname "$CKPT")"
            TMP_DL=$(mktemp -d "$(dirname "$CKPT")/.download.XXXXXX")
            "$PYTHON" - "$WANDB_ENTITY/ViT-perforated/checkpoint-$BASELINE:last" "$TMP_DL" <<'PY'
import os, sys, wandb
path, dest = sys.argv[1:]
artifact = wandb.Api(api_key=os.environ.get('WANDB_API_TOKEN') or None).artifact(path, type='model')
artifact.download(dest)
print(f"=> downloaded {path} (step {artifact.metadata.get('step')}, acc@1 {artifact.metadata.get('acc1')}) to {dest}")
PY
            mv -f "$TMP_DL/checkpoint.pth.tar" "$CKPT"
            rm -rf "$TMP_DL"
        fi
    fi
    test -r "$CKPT" || { echo "cannot read CKPT=$CKPT" >&2; exit 1; }
    RUN_NAME=${PREFIX}$(date +%Y%m%d_%H%M)
    START_ARGS=(--pai-init "$CKPT")
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
    --batch-size 1024 --accum-freq 4 --tf32 --mlp-head --torchvision-inception-crop \
    --epochs 300 --lr 1e-3 --wd 1e-4 --log-steps 250 --workers "$WORKERS" \
    "${START_ARGS[@]}" \
    --pai-layers 11 --pai-target ln_mlp --max-dendrites 2 \
    --pai-patience 2 --pai-n-epochs-to-switch 10 \
    --logs "$LOGS" --name "$RUN_NAME" ${REPORT[@]+"${REPORT[@]}"} 2>&1 | tee -a "$WANDB_DIR/train.log"
