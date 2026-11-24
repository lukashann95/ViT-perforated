#!/bin/bash
# Non-Slurm version of scripts/l40_pai_resume.sbatch for a single-GPU machine (riddle): continue a
# PAI run from its pre-switch checkpoint (--pai-resume), with dendrites or as the matched
# no-dendrite control (CONTROL=1: same tracker, plateau schedule and LR resets at every switch,
# never any dendrites). Same main.py settings as the L40 script, except that the batch of 1024 may
# be accumulated (ACCUM, default 4: 4 x 256, the same neuron-phase update as 1024 x 1).
#   RESUME_WANDB=vit_tiny_pai_scratch_9_10_11_ln_mlp_pb_fp32_20pct_20261120_1256 CONTROL=1 \
#       RUN_SUFFIX=_$USER ./scripts/pai_resume.sh
#   DRY_RUN=1 ...   print the main.py command and exit
#
# Env: RESUME (pre_switch checkpoint.pth.tar with pai/best_model.pt beside it) or RESUME_WANDB
#   (downloads checkpoint-<run>:pre_switch), LAYERS ("9 10 11"), PAI_TARGET (ln_mlp), CONTROL (0),
#   ACCUM (4), LR (1e-3), MAX_DENDRITES (2), PAI_PATIENCE (5), PAI_N_EPOCHS_TO_SWITCH (15),
#   EPOCHS (350, total including the source run's epochs; a cap, PAI ends the run itself),
#   EXTRA_ARGS (extra main.py flags), HISTORY_FROM_SOURCE (0; 1: --wandb-history-from), RUN_SUFFIX, WANDB_CKPTS (1), LOGS (~/runs).
#   DATA, GPU, WORKERS, credentials: scripts/machine_env.sh.
set -euo pipefail
LAYERS=${LAYERS:-9 10 11}
PAI_TARGET=${PAI_TARGET:-ln_mlp}
CONTROL=${CONTROL:-0}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-3}
MAX_DENDRITES=${MAX_DENDRITES:-2}
PAI_PATIENCE=${PAI_PATIENCE:-5}
PAI_N_EPOCHS_TO_SWITCH=${PAI_N_EPOCHS_TO_SWITCH:-15}
EPOCHS=${EPOCHS:-350}
RUN_SUFFIX=${RUN_SUFFIX:-}
WANDB_CKPTS=${WANDB_CKPTS:-1}
read -r -a LAYER_LIST <<< "$LAYERS"
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
source "$REPO/scripts/machine_env.sh"
LOGS=${LOGS:-$HOME/runs}
: "${PAIEMAIL:?PAIEMAIL is not set}"
: "${PAITOKEN:?PAITOKEN is not set}"
cd "$REPO"

if [ -n "${RESUME_WANDB:-}" ]; then
    RESUME=$LOGS/ckpts/${RESUME_WANDB}_pre_switch/checkpoint.pth.tar
    if [ ! -r "$RESUME" ] && [ "${DRY_RUN:-0}" != 1 ]; then
        # Same layout as the source run's archive/pre_switch/: checkpoint.pth.tar and pai/best_model.pt.
        "$PYTHON" - "$RESUME_WANDB" "$(dirname "$RESUME")" <<'PY'
import os, sys, wandb
run, dest = sys.argv[1:]
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/checkpoint-{run}:pre_switch"
artifact = wandb.Api(api_key=os.environ.get('WANDB_API_TOKEN') or None).artifact(path, type='model')
artifact.download(dest)
print(f"=> downloaded {path} (step {artifact.metadata.get('step')}) to {dest}")
PY
    fi
    SRC_RUN=$RESUME_WANDB
else
    : "${RESUME:?set RESUME or RESUME_WANDB}"
    SRC_DIR=${RESUME%%/archive/*}
    [ "$SRC_DIR" = "$RESUME" ] && SRC_DIR=${RESUME%%/checkpoints/*}
    SRC_RUN=$(basename "$SRC_DIR")
fi
if [ "${DRY_RUN:-0}" != 1 ]; then
    test -r "$RESUME" && test -r "$(dirname "$RESUME")/pai/best_model.pt" || {
        echo "RESUME=$RESUME or its pai/best_model.pt is missing" >&2; exit 1; }
fi

PAI_ARGS=(--pai-layers "${LAYER_LIST[@]}")
TAG=$(IFS=-; echo "${LAYER_LIST[*]}")
if [ "$CONTROL" = 1 ]; then PAI_ARGS+=(--pai-control); TAG=${TAG}_control; fi
# HISTORY_FROM_SOURCE=1: copy the source run's W&B history into this run (it must be in WANDB_PROJECT).
HISTORY=()
[ "${HISTORY_FROM_SOURCE:-0}" = 1 ] && HISTORY=(--wandb-history-from "$SRC_RUN")
REPORT=(--report-to wandb)
[ "$WANDB_CKPTS" = 1 ] && REPORT+=(--wandb-checkpoints)
COMMIT=$(git log --oneline -1 2>/dev/null || echo "unknown commit")
RUN_NAME=vit_tiny_pai_resume_${TAG}_${PAI_TARGET}_pb_b1024_acc${ACCUM}_lr${LR}_20pct_$(date +%Y%m%d_%H%M)${RUN_SUFFIX}
export WANDB_DIR=$LOGS/$RUN_NAME

CMD=("$PYTHON" -u main.py "$DATA"
    --gpu 0 --train-fraction 0.2 --subset-seed 42
    --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16
    --batch-size 1024 --accum-freq "$ACCUM" --tf32 --mlp-head --torchvision-inception-crop
    --epochs "$EPOCHS" --lr "$LR" --log-steps 250 --workers "$WORKERS"
    --pai-resume "$RESUME" "${HISTORY[@]}"
    "${PAI_ARGS[@]}" --pai-target "$PAI_TARGET" --max-dendrites "$MAX_DENDRITES"
    --pai-patience "$PAI_PATIENCE" --pai-n-epochs-to-switch "$PAI_N_EPOCHS_TO_SWITCH"
    "${EXTRA[@]}"
    --wandb-notes "Resume of $SRC_RUN pre_switch, layers $LAYERS $PAI_TARGET, CONTROL=$CONTROL, batch 1024 (acc $ACCUM), LR $LR, patience $PAI_PATIENCE, n-epochs-to-switch $PAI_N_EPOCHS_TO_SWITCH, $COMMIT"
    --logs "$LOGS" --name "$RUN_NAME" "${REPORT[@]}")

echo "=== $RUN_NAME from $RESUME, commit $COMMIT"
if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$WANDB_DIR"
"${CMD[@]}" 2>&1 | tee "$WANDB_DIR/train.log"
