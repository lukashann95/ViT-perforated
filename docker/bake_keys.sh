#!/bin/bash
# Build vit-perforated-l40:keys: the vit-perforated-l40 image plus a .env with the W&B and PAI
# keys, so whoever runs it does not need their own.
#
#   docker/bake_keys.sh                              # uses the repo's .env
#   ENV_FILE=/path/to/.env docker/bake_keys.sh
#   docker save vit-perforated-l40:keys | gzip > vit-perforated-l40-keys.tar.gz   # file to hand over
#
# Anyone who has this image can read the keys out of it (docker run ... cat .env), so:
#   - hand the image file only to people you trust with the keys, and never push it to a public
#     registry such as Docker Hub;
#   - every W&B run started from it is logged under the W&B user who owns WANDB_API_TOKEN;
#   - if the image leaks, revoke the W&B key at https://wandb.ai/authorize and ask PerforatedAI
#     for a new token.
# docker/run_l40.sh uses this image automatically when the machine has it.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE=${ENV_FILE:-$REPO/.env}
BASE=${BASE:-vit-perforated-l40}
TAG=${TAG:-vit-perforated-l40:keys}

test -f "$ENV_FILE" || { echo "no $ENV_FILE" >&2; exit 1; }
for var in WANDB_API_TOKEN PAIEMAIL PAITOKEN; do
    grep -qE "^$var=.+" "$ENV_FILE" || { echo "$var is empty or missing in $ENV_FILE" >&2; exit 1; }
done
docker image inspect "$BASE" >/dev/null 2>&1 || {
    echo "no $BASE image: docker build -f docker/Dockerfile -t $BASE ." >&2; exit 1; }

# Build from a context holding only the .env (the repo's .dockerignore keeps .env out of images).
CONTEXT=$(mktemp -d)
trap 'rm -rf "$CONTEXT"' EXIT
cp "$ENV_FILE" "$CONTEXT/.env"
# Readable by every uid: run_l40.sh runs the container as the host user.
docker build -q -t "$TAG" -f - "$CONTEXT" <<EOF
FROM $BASE
COPY --chmod=444 .env /opt/ViT-perforated/.env
EOF
echo "built $TAG (keys from $ENV_FILE)"
echo "to hand it over:  docker save $TAG | gzip > vit-perforated-l40-keys.tar.gz"
echo "on the other machine:  docker load -i vit-perforated-l40-keys.tar.gz"
