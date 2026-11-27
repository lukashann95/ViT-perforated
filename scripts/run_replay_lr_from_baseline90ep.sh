#!/bin/bash
# Control for run_pai_l11_from_baseline90ep.sh: the same ViT-Tiny without PAI or dendrites, trained
# on the LR schedule that PAI run followed (vit_tiny_pai_init_l11_ln_mlp_pb_fp32_b1024_acc4_lr1e-3_
# from_base90ep_20261126_0317), with its warm restarts: at each restart the best model of the cycle
# is reloaded and AdamW starts fresh, as PAI does. Shows how much of that run's gain the restarts
# alone give. The schedule is scripts/replay_lr/<PAI run>.json (see lr_replay.py and
# scripts/build_replay_lr.py): PAI's neuron phases on the path to its final model, without its
# dendrite phases (neurons frozen) and abandoned LR-search trials, so 41250 steps instead of 58250.
# Starts from the final checkpoint of the 90-epoch cosine baseline
# vit_tiny_tf32_1gpu_20pct_90ep_subset42_riddle_20261121_040828 (42.96%), as the PAI run did.
# Batch 1024 as 4 x 256 microbatches, --lr 1e-3 (only sets the weight decay: --wd 1e-4 / 1e-3 =
# 0.1, as in the PAI run), validation every 250 steps, FP32 + TF32.
#
#   GPU=1 ./scripts/run_replay_lr_from_baseline90ep.sh
#   CONTINUE_RUN=<run name> GPU=1 ./scripts/run_replay_lr_from_baseline90ep.sh   # after a crash
#
# The checkpoint is downloaded once from W&B (IdlF26Group40/ViT-perforated, artifact
# checkpoint-<baseline>:last) to $LOGS/ckpts/<baseline>/, where the PAI run's script put it too;
# CKPT=<path> uses a local copy instead.
# LOGS: run directories (default ~/runs). WANDB=0: no W&B.
# DATA, GPU, WORKERS, WANDB_ENTITY / WANDB_PROJECT: see scripts/machine_env.sh
# (WANDB_PROJECT defaults to ViT-perforated-plateau here, next to the PAI run).
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
export WANDB_PROJECT=${WANDB_PROJECT:-ViT-perforated-plateau}
# Sets DATA, WORKERS, PYTHON, the W&B entity and the visible GPU.
source "$REPO/scripts/machine_env.sh"
LOGS=${LOGS:-$HOME/runs}
BASELINE=vit_tiny_tf32_1gpu_20pct_90ep_subset42_riddle_20261121_040828
PAI_RUN=vit_tiny_pai_init_l11_ln_mlp_pb_fp32_b1024_acc4_lr1e-3_from_base90ep_20261126_0317
SCHEDULE=$REPO/scripts/replay_lr/$PAI_RUN.json
PREFIX=vit_tiny_replay_lr_of_pai_init_l11_b1024_acc4_lr1e-3_from_base90ep_

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
    START_ARGS=(--init "$CKPT")
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
    --lr 1e-3 --wd 1e-4 --log-steps 250 --workers "$WORKERS" \
    --replay-lr "$SCHEDULE" "${START_ARGS[@]}" \
    --logs "$LOGS" --name "$RUN_NAME" ${REPORT[@]+"${REPORT[@]}"} 2>&1 | tee -a "$WANDB_DIR/train.log"
