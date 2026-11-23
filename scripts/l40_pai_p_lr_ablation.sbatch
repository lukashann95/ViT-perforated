#!/bin/bash
#SBATCH --job-name=vit-plr-l40
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
# Dendrite-phase LR sweep (scripts/pai_p_lr_ablation.sh) on an L40 / L40S, for both dendrite
# targets in one submission: a Slurm array with one task per target, each on its own GPU (in
# parallel if two GPUs are free, otherwise one after the other):
#   task 0: TARGET=ln_mlp  (block 11's pre-norm and whole MLP)
#   task 1: TARGET=ln_fc1  (block 11's pre-norm and first MLP layer)
# Each task loads the batch-1024 plain baseline from W&B and trains only the first dendrite phase
# for 25 epochs at LR scales 0.01, 0.1, 1, 10 and 100 (roughly 2 h each, ~10 h per task).
#
# For a machine set up with scripts/setup_machine.sh (default settings: .venv and .venv-data in
# the repo, parquet shards in <repo parent>/imagenet_1k_resized_256/data, keys in <repo>/.env).
# Partition and account differ per cluster, so pass them at submission (and e.g. --gres=gpu:l40:1
# if the cluster wants the GPU type named). From the repo root:
#   cd ViT-perforated
#   sbatch -p <partition> -A <account> scripts/l40_pai_p_lr_ablation.sbatch
# Only one target:  sbatch ... --array=1 scripts/l40_pai_p_lr_ablation.sbatch   (1 = ln_fc1)
#
# Each task unpacks the shards to node-local disk ($SLURM_TMPDIR, else /tmp; ~30 GB, deleted at
# the end). Runs go to <repo parent>/runs/vit_tiny_pai_plr_l11_<target>_b1024_acc1_lr1e-3_k<k>_<time>
# and to W&B; Slurm logs to slurm-vit-plr-l40-<job id>_<task>.out where you submitted.
# Other settings (--export=ALL,NAME=value): CKPT_WANDB / CKPT (another starting checkpoint, with its
# PEAK_LR and BATCH), SCALES, ACCUM, LOGS; see scripts/pai_p_lr_ablation.sh.
set -euo pipefail
TARGETS=(ln_mlp ln_fc1)
TASK=${SLURM_ARRAY_TASK_ID:-0}
[ "$TASK" -lt ${#TARGETS[@]} ] || { echo "array task $TASK: only 0 (ln_mlp) and 1 (ln_fc1) exist" >&2; exit 1; }
export TARGET=${TARGET:-${TARGETS[$TASK]}}
# The plain 90-epoch batch-1024 baseline (42.96% val top-1), trained at peak LR 1e-3.
if [ -z "${CKPT:-}" ]; then
    export CKPT_WANDB=${CKPT_WANDB:-vit_tiny_tf32_1gpu_20pct_90ep_subset42_riddle_20261121_040828}
fi
export PEAK_LR=${PEAK_LR:-1e-3} BATCH=${BATCH:-1024} ACCUM=${ACCUM:-1}

# sbatch runs a copy of this script, so the repo is the directory sbatch was called from.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
test -f "$REPO/scripts/pai_p_lr_ablation.sh" || { echo "No $REPO/scripts/pai_p_lr_ablation.sh: submit from the repo root" >&2; exit 1; }
export REPO
export LOGS=${LOGS:-$(dirname "$REPO")/runs}
echo "Array task $TASK: TARGET=$TARGET, from ${CKPT:-W&B run $CKPT_WANDB}, PEAK_LR=$PEAK_LR BATCH=$BATCH ACCUM=$ACCUM"
exec bash "$REPO/scripts/pai_p_lr_ablation.sh"
