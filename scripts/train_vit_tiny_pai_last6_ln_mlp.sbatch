#!/bin/bash
#SBATCH --job-name=vit-pai-last6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

# Slurm wrapper for train_vit_tiny_pai_last6_ln_mlp.sh; all training settings live there.
# Submit from the repo root, adding your cluster's partition/account (and GPU type if needed):
#   sbatch -p <partition> -A <account> scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch
#   sbatch -p <partition> --gres=gpu:l40:1 scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch
# The run script's variables pass through --export, e.g.:
#   sbatch --export=ALL,CONTROL=1 ... scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch
#
# A full run can take longer than the cluster's time limit (check: sinfo -o "%P %l").
# To continue after a timeout, pass the stopped run's name (the log directory under ~/runs):
#   sbatch --export=ALL,CONTINUE_RUN=<run name> ... scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch

set -euo pipefail

# Slurm runs a copy of this file from its spool directory, so locate the repo by where the job
# was submitted from rather than by this file's path.
cd "$SLURM_SUBMIT_DIR"
test -f scripts/train_vit_tiny_pai_last6_ln_mlp.sh || {
    echo "submit from the repo root (no scripts/ in $SLURM_SUBMIT_DIR)" >&2; exit 1; }
exec ./scripts/train_vit_tiny_pai_last6_ln_mlp.sh "$@"
