#!/bin/bash
# Run setup_machine.sh or l40_pai_resume.sbatch inside the vit-perforated-l40 image (docker/Dockerfile).
#
#   docker/run_l40.sh setup [VAR=value ...]   # download the dataset, check keys, GPU smoke test
#   docker/run_l40.sh train [VAR=value ...]   # start a training run in the background
#   docker/run_l40.sh shell                   # a shell in the container, with the same mounts
#
# VAR=value pairs are passed to the script, e.g.
#   docker/run_l40.sh train SCRATCH=1                # fresh run, layers 9, 10 and 11
#   docker/run_l40.sh train ARM=11 RESUME=/data/checkpoints/<run>/archive/<snapshot>/checkpoint.pth.tar
#   docker/run_l40.sh setup DATA_INCLUDE="data/train-00000-* data/val-00000-*"   # quick test
# Paths given to the scripts are container paths: HOST_DATA is mounted at /data.
#
# Settings (environment variables of this script):
#   HOST_DATA=path     host folder for shards, checkpoints and runs (default ~/vit-data):
#                      /data/imagenet_1k_resized_256, /data/checkpoints, /data/runs
#   HOST_UNPACK=path   host folder the training job unpacks ImageNet into (~30 GB, deleted at the
#                      end; default $HOST_DATA/unpack). Put it on a fast local disk.
#   ENV_FILE=path      the .env with WANDB_API_TOKEN, PAIEMAIL, PAITOKEN (default: repo's .env).
#                      Optional with the vit-perforated-l40:keys image (docker/bake_keys.sh),
#                      which has the keys built in; a file given here overrides them.
#   GPU=n              host GPU index to use (default 0); PAI runs need exactly one GPU
#   IMAGE=name         image to run (default vit-perforated-l40:keys if this machine has it,
#                      otherwise vit-perforated-l40)
#   DOCKER_ARGS="..."  extra docker run options, e.g. "--network host" when DNS fails in containers

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CMD=${1:?usage: $0 setup|train|shell [VAR=value ...]}
shift
if [ -z "${IMAGE:-}" ]; then
    IMAGE=vit-perforated-l40
    docker image inspect vit-perforated-l40:keys >/dev/null 2>&1 && IMAGE=vit-perforated-l40:keys
fi
HOST_DATA=${HOST_DATA:-$HOME/vit-data}
HOST_UNPACK=${HOST_UNPACK:-$HOST_DATA/unpack}
ENV_FILE=${ENV_FILE:-$REPO/.env}
GPU=${GPU:-0}

ENV_MOUNT=()
if [ -f "$ENV_FILE" ]; then
    ENV_MOUNT=(-v "$ENV_FILE:/opt/ViT-perforated/.env:ro")
elif ! docker run --rm --entrypoint test "$IMAGE" -f /opt/ViT-perforated/.env; then
    echo "no $ENV_FILE, and $IMAGE has no keys built in: cp $REPO/.env.example $REPO/.env and" \
         "fill in the keys (or set ENV_FILE, or use an image from docker/bake_keys.sh)" >&2
    exit 1
fi
mkdir -p "$HOST_DATA" "$HOST_UNPACK"

ENV_ARGS=()
for kv in "$@"; do
    case "$kv" in
        *=*) ENV_ARGS+=(-e "$kv") ;;
        *) echo "expected VAR=value, got '$kv'" >&2; exit 1 ;;
    esac
done
STAMP=$(date +%Y%m%d_%H%M%S)
TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)
# shellcheck disable=SC2206  # word-split on purpose
RUN_ARGS=(
    ${DOCKER_ARGS:-}
    --gpus "device=$GPU"
    --user "$(id -u):$(id -g)"
    --shm-size=16g    # DataLoader workers pass batches through /dev/shm
    ${ENV_MOUNT[@]+"${ENV_MOUNT[@]}"}
    -v "$HOST_DATA:/data"
    -v "$HOST_UNPACK:/unpack"
    -e "LOCAL=/unpack/imagenet_$STAMP"
    ${ENV_ARGS[@]+"${ENV_ARGS[@]}"}
)

case "$CMD" in
    setup)
        exec docker run --rm ${TTY[@]+"${TTY[@]}"} "${RUN_ARGS[@]}" "$IMAGE" ./scripts/setup_machine.sh ;;
    shell)
        exec docker run --rm ${TTY[@]+"${TTY[@]}"} "${RUN_ARGS[@]}" "$IMAGE" bash ;;
    train)
        NAME=vit-pai-$STAMP
        LOG=/data/runs/docker-$STAMP.out
        mkdir -p "$HOST_DATA/runs"
        docker run -d --name "$NAME" "${RUN_ARGS[@]}" "$IMAGE" \
            bash -c "set -o pipefail; bash scripts/l40_pai_resume.sbatch 2>&1 | tee $LOG" >/dev/null
        echo "started container $NAME from $IMAGE"
        echo "  follow:  docker logs -f $NAME      (also in $HOST_DATA/runs/docker-$STAMP.out)"
        echo "  stop:    docker stop $NAME"
        echo "  remove:  docker rm $NAME           (after it has finished)" ;;
    *)
        echo "usage: $0 setup|train|shell [VAR=value ...]" >&2; exit 1 ;;
esac
