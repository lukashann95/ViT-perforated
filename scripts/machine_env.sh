# Machine setup shared by the launchers (run_vit_tiny_baseline.sh, pai_p_lr_ablation.sh), which
# source it after setting REPO. The same launcher then runs on riddle / the 3090s (./scripts/...)
# and as a Slurm job on NCSA Delta (sbatch scripts/..., from the repo root).
#
# Sets, each only if not set already:
#   DATA     ImageFolder root: ~/data/imagenet or $DATA_DIR/imagenet if one exists. Otherwise (and always on Delta) the
#            parquet shards in PARQUET_DIR are unpacked to LOCAL (~30 GB, deleted when the script
#            exits): PARQUET_DIR defaults to Delta's shared copy, elsewhere to where
#            scripts/setup_machine.sh downloads them ($DATA_DIR/imagenet_1k_resized_256/data,
#            DATA_DIR = the repo's parent); LOCAL to ${SLURM_TMPDIR:-/tmp/$USER}/imagenet_<job id>.
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

if [ -z "${DATA:-}" ] && [ "$ON_DELTA" = 0 ]; then
    # An unpacked ImageFolder: riddle / the 3090s, or setup_machine.sh UNPACK=1 ($DATA_DIR/imagenet).
    for dir in "$HOME/data/imagenet" "${DATA_DIR:-$(dirname "$REPO")}/imagenet"; do
        if [ -d "$dir/train" ] && [ -d "$dir/val" ]; then DATA=$dir; break; fi
    done
fi
if [ -z "${DATA:-}" ]; then
    # 1.3M small files would be slow on shared storage, so unpack the parquet shards to local disk.
    if [ "$ON_DELTA" = 1 ]; then
        PARQUET_DIR=${PARQUET_DIR:-/projects/bjto/shared/imagenet_1k_resized_256/data}
    else
        PARQUET_DIR=${PARQUET_DIR:-${DATA_DIR:-$(dirname "$REPO")}/imagenet_1k_resized_256/data}
    fi
    if ! compgen -G "$PARQUET_DIR/train-*.parquet" >/dev/null || ! compgen -G "$PARQUET_DIR/val-*.parquet" >/dev/null; then
        echo "No ImageNet: no ~/data/imagenet and no train-*/val-*.parquet shards in $PARQUET_DIR" \
             "(set DATA to an ImageFolder or PARQUET_DIR to the shards; scripts/setup_machine.sh downloads them)" >&2
        exit 1
    fi
    DATA=${LOCAL:-${SLURM_TMPDIR:-/tmp/${USER:-$(id -un)}}/imagenet_${SLURM_JOB_ID:-$$}}
    mkdir -p "$DATA"
    avail_gb=$(df -BG --output=avail "$DATA" | tail -1 | tr -dc 0-9)
    if [ "$avail_gb" -lt 40 ]; then
        echo "only ${avail_gb} GB free on $(df --output=target "$DATA" | tail -1); need ~30 GB (set LOCAL)" >&2
        exit 1
    fi
    trap 'rm -rf "$DATA"' EXIT
    echo "Unpacking ImageNet from $PARQUET_DIR to $DATA ($avail_gb GB free)..."
    start=$SECONDS
    # pyarrow lives in a small separate venv so the training lockfile stays unchanged: the one
    # delta_pai_resume.sbatch uses on Delta, else setup_machine.sh's default.
    if [ -z "${DATA_VENV:-}" ]; then
        DATA_VENV=$REPO/.venv-data
        if [ "$ON_DELTA" = 1 ] && [ -x "$DELTA_PROJECT/.venv-data/bin/python" ]; then
            DATA_VENV=$DELTA_PROJECT/.venv-data
        fi
    fi
    [ -x "$DATA_VENV/bin/python" ] || { echo "No pyarrow venv at $DATA_VENV: run scripts/setup_machine.sh" >&2; exit 1; }
    "$DATA_VENV/bin/python" "$REPO/scripts/parquet_to_imagefolder.py" "$PARQUET_DIR" "$DATA" \
        --workers "${SLURM_CPUS_PER_TASK:-$(env -u OMP_NUM_THREADS nproc)}"
    echo "Unpacked in $((SECONDS - start)) s"
fi
DATA=${DATA:-$HOME/data/imagenet}
test -d "$DATA/train" && test -d "$DATA/val" || { echo "No train/ and val/ in $DATA" >&2; exit 1; }
