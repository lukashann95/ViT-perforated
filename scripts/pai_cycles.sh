#!/bin/bash
# Full Perforated Backpropagation run from a plain (non-PAI) ViT-Tiny checkpoint, with fixed-length
# phases: perforate blocks LAYERS (TARGET), then alternate
#     dendrite (p) phase, P_EPOCHS  ->  neuron (n) phase, N_EPOCHS  ->  p  ->  n  ...
# for MAX_DENDRITES rounds. --pai-init forces the first switch at step 0, so the run starts with a
# dendrite phase. PAI ends the run itself (main.py stops on its training_complete):
#   - after the n phase of round MAX_DENDRITES, if that n phase set a new best validation score;
#   - earlier, if an n phase did NOT set a new best: PAI retries that round once from the previous
#     best model (max_dendrite_tries 2), and after a second failure stops and keeps best_model.
# --max-steps (all rounds plus one epoch) is only a backstop.
#
# Batches: --batch-size BATCH --accum-freq ACCUM. Neuron phases accumulate ACCUM microbatches of
# BATCH/ACCUM into one step of BATCH; dendrite phases step on every microbatch (no accumulation).
# Defaults: neurons on 1024 (4 x 256), dendrites on 256; neuron LR 1e-3 (the b1024 baseline's
# peak), dendrite LR 1e-3 / sqrt(4) = 5e-4; clean (un-augmented) data and the loss scaled by the
# microbatch size in dendrite phases (both gave layer-11 ln_fc1 PB scores of ~0.14 on riddle).
#   CKPT=/path/checkpoint.pth.tar RUN_SUFFIX=_$USER ./scripts/pai_cycles.sh
#   CKPT_WANDB=<run name> CKPT_VERSION=v6 ./scripts/pai_cycles.sh   # downloads checkpoint-<name>:v6
#   DRY_RUN=1 ...   print the main.py command and exit
#
# Env: CKPT or CKPT_WANDB (required; CKPT_VERSION default last); BATCH (1024), ACCUM (4), LR (1e-3),
#   P_LR_SCALING (sqrt: dendrite LR = LR / sqrt(ACCUM); linear, none), WD (1e-4, the baseline's, neuron phases only;
#   dendrite phases always use 0), P_EPOCHS (25), N_EPOCHS (50),
#   MAX_DENDRITES (2), LAYERS ("9 10 11"), TARGET (ln_fc1), CLEAN (1), P_LOSS_SCALE (BATCH / ACCUM),
#   RUN_SUFFIX (none), WANDB (1), WANDB_CKPTS (1: --wandb-checkpoints), LOGS (~/runs),
#   CONTROL (0; 1: matched no-dendrite control, --pai-control: same LR resets every N_EPOCHS, no
#   dendrite phases), EXTRA_ARGS (extra main.py flags, word-split).
#   DATA, GPU, WORKERS, W&B and PAI credentials: see scripts/machine_env.sh.
#
# LR schedule: PAI rebuilds a cosine from the phase's starting LR at every phase, with one T_max for
# all phases (--pai-cosine-epochs). It is N_EPOCHS, so each neuron phase decays to 0 at its end and
# each dendrite phase (shorter) only part of the way, which is fine: PB scores level off within
# 2-6 epochs. A phase longer than T_max would see CosineAnnealingLR climb back up.
set -euo pipefail
[ -n "${CKPT:-}${CKPT_WANDB:-}" ] || { echo "set CKPT to a plain main.py checkpoint, or CKPT_WANDB to its run name" >&2; exit 1; }
BATCH=${BATCH:-1024}
ACCUM=${ACCUM:-4}
LR=${LR:-1e-3}
P_LR_SCALING=${P_LR_SCALING:-sqrt}
WD=${WD:-1e-4}
P_EPOCHS=${P_EPOCHS:-25}
N_EPOCHS=${N_EPOCHS:-50}
MAX_DENDRITES=${MAX_DENDRITES:-2}
LAYERS=${LAYERS:-9 10 11}
TARGET=${TARGET:-ln_fc1}
CLEAN=${CLEAN:-1}
P_LOSS_SCALE=${P_LOSS_SCALE:-$((BATCH / ACCUM))}
RUN_SUFFIX=${RUN_SUFFIX:-}
WANDB_CKPTS=${WANDB_CKPTS:-1}
CONTROL=${CONTROL:-0}
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
if [ "$P_EPOCHS" -gt "$N_EPOCHS" ]; then echo "P_EPOCHS > N_EPOCHS: the cosine (T_max = N_EPOCHS) would rise again" >&2; exit 1; fi
read -r -a LAYER_LIST <<< "$LAYERS"

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
# Sets DATA, WORKERS, PYTHON, the PAI credentials, the W&B entity/project and the visible GPU.
source "$REPO/scripts/machine_env.sh"
LOGS=${LOGS:-$HOME/runs}
: "${PAIEMAIL:?PAIEMAIL is not set}"
: "${PAITOKEN:?PAITOKEN is not set}"
cd "$REPO"

if [ -z "${CKPT:-}" ]; then
    CKPT_VERSION=${CKPT_VERSION:-last}
    CKPT=$LOGS/ckpts/${CKPT_WANDB}_${CKPT_VERSION}/checkpoint.pth.tar
    if [ ! -r "$CKPT" ] && [ "${DRY_RUN:-0}" != 1 ]; then
        mkdir -p "$(dirname "$CKPT")"
        TMP_DL=$(mktemp -d "$(dirname "$CKPT")/.download.XXXXXX")
        "$PYTHON" - "$CKPT_WANDB" "$CKPT_VERSION" "$TMP_DL" <<'PY'
import os, sys, wandb
run, version, dest = sys.argv[1:]
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/checkpoint-{run}:{version}"
artifact = wandb.Api(api_key=os.environ.get('WANDB_API_TOKEN') or None).artifact(path, type='model')
artifact.download(dest)
print(f"=> downloaded {path} (step {artifact.metadata.get('step')}, "
      f"acc@1 {artifact.metadata.get('acc1')}) to {dest}")
PY
        mv -f "$TMP_DL/checkpoint.pth.tar" "$CKPT"
        rm -rf "$TMP_DL"
    fi
fi
[ "${DRY_RUN:-0}" = 1 ] || test -r "$CKPT" || { echo "cannot read CKPT=$CKPT" >&2; exit 1; }

# One validation per epoch of the 20% subset (256,190 images), so PAI's epochs are real epochs.
LOG_STEPS=$((256190 / BATCH))
TOTAL_EPOCHS=$((MAX_DENDRITES * (P_EPOCHS + N_EPOCHS) + 1))   # +1: PAI stops first
if [ "$CONTROL" = 1 ]; then TOTAL_EPOCHS=$((MAX_DENDRITES * N_EPOCHS + 1)); fi
MAX_STEPS=$((TOTAL_EPOCHS * LOG_STEPS))
P_ARGS=()
TAG=
if [ "$CONTROL" = 1 ]; then
    P_ARGS+=(--pai-control); TAG=_ctrl
else
    if [ "$CLEAN" = 1 ]; then P_ARGS+=(--pai-p-clean-data); TAG=_clean; fi
    if [ "$P_LOSS_SCALE" != 1 ]; then P_ARGS+=(--pai-p-loss-scale "$P_LOSS_SCALE"); TAG=${TAG}_ls${P_LOSS_SCALE}; fi
fi
REPORT=()
if [ "${WANDB:-1}" = 1 ]; then
    REPORT=(--report-to wandb)
    if [ "$WANDB_CKPTS" = 1 ]; then REPORT+=(--wandb-checkpoints); fi
fi
LAYER_TAG=$(IFS=-; echo "${LAYER_LIST[*]}")
COMMIT=$(git log --oneline -1 2>/dev/null || echo "unknown commit (no .git)")
RUN_NAME=vit_tiny_pai_cyc_pb${TAG}_l${LAYER_TAG}_${TARGET}_b${BATCH}_acc${ACCUM}_lr${LR}_p${P_EPOCHS}_n${N_EPOCHS}_d${MAX_DENDRITES}_$(date +%Y%m%d_%H%M)${RUN_SUFFIX}
export WANDB_DIR=$LOGS/$RUN_NAME

CMD=("$PYTHON" -u main.py "$DATA"
    --gpu 0 --train-fraction 0.2 --subset-seed 42
    --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16
    --batch-size "$BATCH" --accum-freq "$ACCUM" --tf32 --mlp-head --torchvision-inception-crop
    --epochs "$TOTAL_EPOCHS" --max-steps "$MAX_STEPS" --log-steps "$LOG_STEPS"
    --lr "$LR" --wd "$WD" --workers "$WORKERS"
    --pai-init "$CKPT" --pai-layers "${LAYER_LIST[@]}" --pai-target "$TARGET" --max-dendrites "$MAX_DENDRITES"
    --pai-fixed-switch "$N_EPOCHS" --pai-p-epochs-to-switch "$P_EPOCHS"
    --pai-schedule cosine --pai-cosine-epochs "$N_EPOCHS" --pai-p-lr-scaling "$P_LR_SCALING"
    "${P_ARGS[@]}" "${EXTRA[@]}"
    --wandb-notes "PB cycles$TAG from $CKPT: ${MAX_DENDRITES} x (p ${P_EPOCHS} + n ${N_EPOCHS} epochs), layers $LAYERS $TARGET, batch $BATCH (neurons; dendrites $((BATCH / ACCUM))), LR $LR (p-phase scaling $P_LR_SCALING), wd $WD, $COMMIT"
    --logs "$LOGS" --name "$RUN_NAME" "${REPORT[@]}")

echo "=== $RUN_NAME: up to $TOTAL_EPOCHS epochs ($MAX_STEPS steps), from $CKPT, commit $COMMIT"
if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$WANDB_DIR"
"${CMD[@]}" 2>&1 | tee "$WANDB_DIR/train.log"
