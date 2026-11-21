#!/bin/bash
# Dendrite-phase LR ablation for one plain (non-PAI) ViT-Tiny checkpoint: for each LR scale K,
# perforate the MLP of block 11 and train only the first Perforated Backpropagation dendrite (p)
# phase for 25 epochs, cosine from K x PEAK_LR down to 0, with no weight decay. Look at PAI's
# PB-score plots (<run>/pai_images, W&B), not validation accuracy. The runs go one after another:
#   CKPT=/path/model_best.pth.tar PEAK_LR=1e-3 BATCH=1024 scripts/pai_p_lr_ablation.sh
# As one task-spooler job on riddle GPU 1:
#   forgeq-gpu 1 -L plr env GPU=1 CKPT=... PEAK_LR=2.5e-4 BATCH=256 ~/ViT-perforated/scripts/pai_p_lr_ablation.sh
#
# PEAK_LR is the neuron run's peak LR (the start of its cosine). The four setups:
#   1024 in one pass:            PEAK_LR=1e-3   BATCH=1024
#   1024 neurons, 256 dendrites: PEAK_LR=1e-3   BATCH=1024 ACCUM=4  (p phase steps per 256 microbatch)
#   256, linear LR scaling:      PEAK_LR=2.5e-4 BATCH=256
#   256, sqrt LR scaling:        PEAK_LR=5e-4   BATCH=256
# Env: CKPT, PEAK_LR, BATCH (required); ACCUM (default 1); SCALES (default "0.01 0.1 1 10 100");
# P_EPOCHS (default 25); LAYER (default 11); DATA (ImageFolder root, default ~/data/imagenet);
# LOGS (default ~/runs); GPU (default 0); WORKERS (default 14); WANDB=0 to skip W&B.
# Needs PAIEMAIL/PAITOKEN (environment, ~/.pai_env or the repo's .env); W&B reads
# WANDB_API_TOKEN from .env.
set -euo pipefail
: "${CKPT:?set CKPT to a plain main.py checkpoint}"
: "${PEAK_LR:?set PEAK_LR to the peak LR of the neuron run}"
: "${BATCH:?set BATCH to 1024 or 256}"
ACCUM=${ACCUM:-1}
SCALES=${SCALES:-0.01 0.1 1 10 100}
P_EPOCHS=${P_EPOCHS:-25}
LAYER=${LAYER:-11}
DATA=${DATA:-$HOME/data/imagenet}
LOGS=${LOGS:-$HOME/runs}
GPU=${GPU:-0}
WORKERS=${WORKERS:-14}
test -r "$CKPT"

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
if [ -f "$HOME/.pai_env" ]; then set -a; source "$HOME/.pai_env"; set +a; fi
if [ -f .env ]; then set -a; source .env; set +a; fi
: "${PAIEMAIL:?PAIEMAIL is not set}"
: "${PAITOKEN:?PAITOKEN is not set}"
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU
python=${PYTHON:-$REPO/.venv/bin/python}
REPORT=()
if [ "${WANDB:-1}" = 1 ]; then REPORT=(--report-to wandb); fi

# One validation per epoch of the 20% subset (256,190 images), so PAI's epochs are real epochs;
# stop right after the P_EPOCHS-th validation, where the fixed switch ends the p phase.
LOG_STEPS=$((256190 / BATCH))
MAX_STEPS=$((P_EPOCHS * LOG_STEPS))
STAMP=$(date +%Y%m%d_%H%M)
FAILED=()
for K in $SCALES; do
    LR=$(awk -v a="$PEAK_LR" -v k="$K" 'BEGIN { printf "%.6g", a * k }')
    RUN_NAME=vit_tiny_pai_plr_l${LAYER}_b${BATCH}_acc${ACCUM}_k${K}_${STAMP}
    export WANDB_DIR=$LOGS/$RUN_NAME
    mkdir -p "$WANDB_DIR"
    echo "=== $RUN_NAME: LR $LR ($K x $PEAK_LR) -> 0 over $P_EPOCHS epochs, from $CKPT," \
         "commit $(git log --oneline -1)"
    # --pai-init perforates the loaded model and forces the switch into the p phase at step 0.
    # With --pai-fixed-switch the p phase lasts exactly --pai-p-epochs-to-switch validations.
    # Neurons are frozen in the p phase, so --wd 0 means no weight decay on the candidates.
    if ! "$python" -u main.py "$DATA" \
        --gpu 0 --train-fraction 0.2 --subset-seed 42 \
        --hidden-dim 192 --num-layers 12 --num-heads 3 --patch-size 16 \
        --batch-size "$BATCH" --accum-freq "$ACCUM" --tf32 --mlp-head --torchvision-inception-crop \
        --epochs "$P_EPOCHS" --max-steps "$MAX_STEPS" --log-steps "$LOG_STEPS" \
        --lr "$LR" --wd 0 --workers "$WORKERS" \
        --pai-init "$CKPT" --pai-layers "$LAYER" --pai-target ln_mlp --max-dendrites 1 \
        --pai-schedule cosine --pai-cosine-epochs "$P_EPOCHS" \
        --pai-fixed-switch 1000 --pai-p-epochs-to-switch "$P_EPOCHS" \
        --wandb-notes "p-phase LR ablation from $CKPT, K=$K, peak LR $PEAK_LR" \
        --logs "$LOGS" --name "$RUN_NAME" "${REPORT[@]}" 2>&1 | tee "$WANDB_DIR/train.log"; then
        FAILED+=("$K")   # e.g. K=100 diverging; keep going with the other scales
    fi
done
if [ ${#FAILED[@]} -gt 0 ]; then echo "Failed scales: ${FAILED[*]}" >&2; exit 1; fi
