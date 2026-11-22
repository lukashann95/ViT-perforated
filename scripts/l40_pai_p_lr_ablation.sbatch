#!/bin/bash
#SBATCH --job-name=vit-plr-l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
# Dendrite-phase LR sweep (scripts/pai_p_lr_ablation.sh) as a Slurm job on an L40 / L40S, for a
# machine set up with scripts/setup_machine.sh (default settings: .venv and .venv-data in the
# repo, parquet shards in <repo parent>/imagenet_1k_resized_256/data, keys in <repo>/.env). The job
# unpacks the shards to node-local disk once ($SLURM_TMPDIR, else /tmp) and runs the five LR
# scales one after another. Partition and account differ per cluster: pass them at submission
# (-p, -A, and e.g. --gres=gpu:l40:1 if the cluster wants the GPU type named). Submit from the
# repo root, e.g. for the batch-1024 baseline with dendrites on block 11's pre-norm and first
# MLP layer:
#   cd ViT-perforated
#   sbatch -p <partition> -A <account> \
#     --export=ALL,CKPT_WANDB=vit_tiny_tf32_1gpu_20pct_90ep_subset42_riddle_20261121_040828,PEAK_LR=1e-3,BATCH=1024,TARGET=ln_fc1 \
#     scripts/l40_pai_p_lr_ablation.sbatch
#
# Settings (--export): CKPT_WANDB=<run name> (downloads checkpoint-<run>:last from W&B) or
# CKPT=<checkpoint file>; PEAK_LR and BATCH of the run the checkpoint comes from; TARGET (ln_mlp,
# the default, or ln_fc1); SCALES (default "0.01 0.1 1 10 100"); ACCUM (default 1: batch 1024 fits
# the L40's 48 GB in one pass). The full list is in scripts/pai_p_lr_ablation.sh. Runs go to
# <repo parent>/runs (LOGS), the Slurm log to slurm-vit-plr-l40-<job id>.out where you submitted.
# Each scale takes roughly 2 h (25 epochs of the 20% subset), so about 10 h for all five; if the
# partition allows less, pass e.g. SCALES="0.01 0.1" and SCALES="1 10 100" as two jobs.
set -euo pipefail
# sbatch runs a copy of this script, so the repo is the directory sbatch was called from.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}}
test -f "$REPO/scripts/pai_p_lr_ablation.sh" || { echo "No $REPO/scripts/pai_p_lr_ablation.sh: submit from the repo root" >&2; exit 1; }
export REPO
export LOGS=${LOGS:-$(dirname "$REPO")/runs}
exec bash "$REPO/scripts/pai_p_lr_ablation.sh"
