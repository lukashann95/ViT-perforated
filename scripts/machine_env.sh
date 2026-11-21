# Machine setup shared by the launchers (run_vit_tiny_baseline.sh, pai_p_lr_ablation.sh), which
# source it after setting REPO. The same launcher then runs on riddle / the 3090s (./scripts/...)
# and as a Slurm job on NCSA Delta (sbatch scripts/..., from the repo root).
#
# Sets, each only if not set already:
#   DATA     ImageFolder root. Delta: the shared parquet shards unpacked to node-local /tmp
#            (needs ~30 GB; deleted when the job exits). Elsewhere: ~/data/imagenet.
#   WORKERS  data-loading workers: Slurm CPUs - 2, elsewhere CPU cores - 1
#   LOGS     Delta only: /projects/bjto/$USER/runs (the launchers have their own defaults)
#   PYTHON   $REPO/.venv/bin/python (after a uv.lock change, run uv sync --frozen)
#   WANDB_ENTITY / WANDB_PROJECT   IdlF26Group40 / ViT-perforated
# Loads KEY=value lines from $REPO/.env, then ~/.pai_env (Delta: /projects/bjto/$USER/.pai_env),
# without overriding variables that are already set, so `WANDB_PROJECT=x ./scripts/...` wins over
# .env (main.py's load_dotenv works the same way). Exports CUDA_VISIBLE_DEVICES=${GPU:-0}, except
# under Slurm, which assigns the GPU itself; and OMP_NUM_THREADS=1.

: "${REPO:?set REPO before sourcing machine_env.sh}"

_load_env_file() {  # KEY=value lines, optionally "export KEY=value"; set variables win
    local line key val
    [ -f "$1" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%$'\r'}
        line=${line#export }
        [[ $line =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
        key=${BASH_REMATCH[1]}
        val=${BASH_REMATCH[2]}
        if [[ $val =~ ^\"(.*)\"$ || $val =~ ^\'(.*)\'$ ]]; then val=${BASH_REMATCH[1]}; fi
        [ -n "${!key+x}" ] || export "$key=$val"
    done < "$1"
}

ON_DELTA=0
if [ -n "${SLURM_JOB_ID:-}" ] && [ -d /projects/bjto ]; then ON_DELTA=1; fi

_load_env_file "$REPO/.env"
if [ "$ON_DELTA" = 1 ]; then
    umask 002   # /projects/bjto is shared with the group
    DELTA_PROJECT=${DELTA_PROJECT:-/projects/bjto/$USER}
    _load_env_file "$DELTA_PROJECT/.pai_env"
    LOGS=${LOGS:-$DELTA_PROJECT/runs}
else
    _load_env_file "$HOME/.pai_env"
fi
export WANDB_ENTITY=${WANDB_ENTITY:-IdlF26Group40} WANDB_PROJECT=${WANDB_PROJECT:-ViT-perforated}

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # Only show one GPU (with 2+ visible GPUs main.py takes its multi-GPU path and crashes).
    export CUDA_VISIBLE_DEVICES=${GPU:-0}
fi
export OMP_NUM_THREADS=1
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    WORKERS=${WORKERS:-$((SLURM_CPUS_PER_TASK - 2))}
else
    # nproc honours OMP_NUM_THREADS, so unset it to get the real core count.
    WORKERS=${WORKERS:-$(( $(env -u OMP_NUM_THREADS nproc) - 1 ))}
fi
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
[ -x "$PYTHON" ] || { echo "No $PYTHON: run uv sync --frozen in $REPO" >&2; exit 1; }

if [ -z "${DATA:-}" ] && [ "$ON_DELTA" = 1 ]; then
    # 1.3M small files would be slow on /projects, so unpack the parquet shards to local disk.
    DATA=/tmp/$USER/imagenet_$SLURM_JOB_ID
    mkdir -p "$DATA"
    avail_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
    if [ "$avail_gb" -lt 40 ]; then
        echo "only ${avail_gb} GB free on $(df --output=target "$DATA" | tail -1); need ~30 GB" >&2
        exit 1
    fi
    trap 'rm -rf "$DATA"' EXIT
    echo "Unpacking ImageNet to $DATA ($avail_gb GB free)..."
    start=$SECONDS
    # pyarrow lives in a small separate venv so the training lockfile stays unchanged: the one
    # delta_pai_resume.sbatch uses, else setup_machine.sh's default.
    if [ -z "${DATA_VENV:-}" ]; then
        DATA_VENV=$DELTA_PROJECT/.venv-data
        [ -x "$DATA_VENV/bin/python" ] || DATA_VENV=$REPO/.venv-data
    fi
    "$DATA_VENV/bin/python" "$REPO/scripts/parquet_to_imagefolder.py" \
        /projects/bjto/shared/imagenet_1k_resized_256/data "$DATA" --workers "$SLURM_CPUS_PER_TASK"
    echo "Unpacked in $((SECONDS - start)) s"
fi
DATA=${DATA:-$HOME/data/imagenet}
test -d "$DATA/train" && test -d "$DATA/val" || { echo "No train/ and val/ in $DATA" >&2; exit 1; }
