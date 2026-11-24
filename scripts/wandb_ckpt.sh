# Sourced by the launchers (after scripts/machine_env.sh, which sets PYTHON, WANDB_ENTITY and WANDB_PROJECT).
# fetch_wandb_ckpt RUN DEST: unless DEST is readable, download checkpoint.pth.tar from the W&B artifact
# checkpoint-RUN:last (written by runs with --wandb-checkpoints) to DEST. It downloads next to DEST and
# renames, so parallel jobs never read a half-written file.
fetch_wandb_ckpt() {
    local run=$1 dest=$2 tmp
    [ -r "$dest" ] && return 0
    mkdir -p "$(dirname "$dest")"
    tmp=$(mktemp -d "$(dirname "$dest")/.download.XXXXXX")
    "$PYTHON" - "$run" "$tmp" <<'PY' || { rm -rf "$tmp"; return 1; }
import os, sys, wandb
run, dest = sys.argv[1:]
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}/checkpoint-{run}:last"
artifact = wandb.Api(api_key=os.environ.get('WANDB_API_TOKEN') or None).artifact(path, type='model')
artifact.download(dest)
print(f"=> downloaded {path} (step {artifact.metadata.get('step')}, "
      f"acc@1 {artifact.metadata.get('acc1')}) to {dest}")
PY
    mv -f "$tmp/checkpoint.pth.tar" "$dest"
    rm -rf "$tmp"
}
