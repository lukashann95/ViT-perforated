# Running the 6-layer PerforatedAI experiment on a new GPU machine

This sets up a machine and trains ViT-Tiny/16 on a fixed 20% subset of ImageNet-1k, with
PerforatedAI dendrites on **encoder layers 6–11**. In each of those layers, the pre-MLP
LayerNorm (`ln_2`) and the whole MLP (Linear → GELU → Linear) are perforated as one unit
(`--pai-target ln_mlp`), giving 6 dendrite modules. Layers 0–5 are tracked but get no dendrites.

| Script | Purpose |
|---|---|
| `scripts/setup_machine.sh` | One-time setup: uv, Python venv, credentials check, ImageNet, smoke tests |
| `scripts/smoke_test.py` | GPU, credentials, layer check, dataset check (setup runs it) |
| `scripts/train_vit_tiny_pai_last6_ln_mlp.sh` | The training run. Holds all settings; run it directly on a personal GPU |
| `scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch` | Slurm wrapper around the `.sh` for clusters |

Target GPUs: RTX 3090 (24 GB), L40 / L40S (48 GB), A40 on Delta (48 GB); the locked PyTorch
(2.6, CUDA 12.4) supports all of them. The scripts were tested end to end on an RTX 5060
(Blackwell), for which setup installs a CUDA 12.8 build of PyTorch instead.

## 1. Get the code

```bash
git clone git@github.com:lukashann95/ViT-perforated.git
cd ViT-perforated
git checkout perforate-6-11-layers-and-general-setup-script
```

If the machine has no access to the GitHub repo, copy the folder over instead (without
`.venv`), e.g. `rsync -a --exclude .venv ViT-perforated/ <user>@<host>:ViT-perforated/`.

## 2. Credentials: the `.env` file

The runs need a W&B API key and a PerforatedAI license. They go in a file named **`.env`** in
the repo root (`ViT-perforated/.env`).

- **`.env`** holds your real keys. It is in `.gitignore`, so it is never committed or pushed,
  and it does **not** come with `git clone`: every machine needs its own copy.
- **`.env.example`** is only the committed template. **Never put keys in it**: anything in it can
  end up on GitHub. (`setup_machine.sh` warns if it finds keys there.)

`.env` must look like this, one `KEY=value` per line, no quotes, no spaces around `=`:

```bash
WANDB_API_TOKEN=<your W&B API key, from https://wandb.ai/authorize>
WANDB_ENTITY=IdlF26Group40
WANDB_PROJECT=ViT-perforated
PAIEMAIL=<PerforatedAI license email>
PAITOKEN=<PerforatedAI license token>
```

Get it onto a machine in one of two ways:

- **Create it on the machine:**
  ```bash
  cd ViT-perforated
  cp .env.example .env && chmod 600 .env     # chmod: readable only by you
  nano .env                                  # fill in WANDB_API_TOKEN, PAIEMAIL, PAITOKEN
  ```
  If you skip this, `setup_machine.sh` creates `.env` for you, stops at step 3, and lists the
  keys still missing. Fill them in and re-run it; finished steps are skipped.
- **Copy a filled-in `.env` from your own computer** (`scp` sends it over SSH, not via GitHub):
  ```bash
  scp .env <user>@<host>:<path>/ViT-perforated/.env
  ssh <user>@<host> chmod 600 <path>/ViT-perforated/.env
  ```

Share the keys only over private channels (not Slack channels, issues or commits). Anyone with the
W&B key can log runs as you; revoke it at https://wandb.ai/authorize when you no longer need it.

## 3. Set up and run

### RTX 3090 (or any personal GPU, no Slurm)

Needs an NVIDIA driver ≥ 550, ~60 GB free disk, internet access, and `tmux`.

```bash
DELETE_PARQUET=1 ./scripts/setup_machine.sh                      # ~40 min
tmux new -s run "./scripts/train_vit_tiny_pai_last6_ln_mlp.sh 2>&1 | tee ~/pai_last6.log"
```

Setup downloads ImageNet (train + val, ~22 GB) and unpacks it to `~/data/imagenet` (~25 GB);
`DELETE_PARQUET=1` removes the downloaded copy afterwards. It ends with `SMOKE TEST PASSED` and
`PAI training run ok`. Detach from tmux with `Ctrl-b d`, reattach with `tmux attach -t run`.

### Slurm clusters: L40 / L40S

Login nodes usually have internet but no GPU, so setup runs in two steps. Find your partition
and account with `sinfo -s` and `sacctmgr show assoc user=$USER format=account,partition`.

```bash
# 1. login node: venv, credentials, dataset (skips the smoke test: no GPU)
DELETE_PARQUET=1 ./scripts/setup_machine.sh
# 2. inside a GPU job: finishes with the smoke test (finished steps are skipped)
srun -p <partition> -A <account> --gpus=1 --cpus-per-task=8 --time=1:00:00 --pty ./scripts/setup_machine.sh
# 3. submit the run from the repo root
sbatch -p <partition> -A <account> scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch
```

If the partition mixes GPU types, request the L40 explicitly, e.g. `--gres=gpu:l40s:1`
(the exact name is in `sinfo -o "%P %G"`). The job writes `slurm-vit-pai-last6-<jobid>.out` in
the directory you submitted from; `squeue -u $USER` shows it, `scancel <jobid>` stops it.

If your home directory has a small quota, put the data and runs elsewhere: add
`DATA_DIR=/path/with/space` to the setup command and pass `IMAGENET=/path/with/space/imagenet`
and `LOG_ROOT=/path/with/space/runs` to `sbatch --export=ALL,IMAGENET=...,LOG_ROOT=...`.

### NCSA Delta (A40, Slurm)

Same two-step Slurm flow, with Delta's account and partition (as in
`scripts/delta_pai_resume.sbatch`). Keep the repo, data and runs under the project directory,
and unpack ImageNet from the parquet shards already shared there instead of downloading again:

```bash
cd /projects/bjto/$USER
git clone git@github.com:lukashann95/ViT-perforated.git && cd ViT-perforated
git checkout perforate-6-11-layers-and-general-setup-script
# copy .env here (step 2), then:
DATA_DIR=/projects/bjto/$USER/data PARQUET_DIR=/projects/bjto/shared/imagenet_1k_resized_256/data \
    ./scripts/setup_machine.sh
srun -A bjto-delta-gpu -p gpuA40x4 --gpus-per-node=1 --cpus-per-task=16 --mem=64G --time=1:00:00 \
    --pty env DATA_DIR=/projects/bjto/$USER/data ./scripts/setup_machine.sh
sbatch -A bjto-delta-gpu -p gpuA40x4 --gpus-per-node=1 --cpus-per-task=16 \
    --export=ALL,IMAGENET=/projects/bjto/$USER/data/imagenet,LOG_ROOT=/projects/bjto/$USER/runs \
    scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch
```

`delta_pai_resume.sbatch` instead unpacks ImageNet to node-local disk at the start of every job,
because reading 1.3M small files from `/projects` can be slow. If the `Data` time in the training
log stays high, that approach is the faster option.

## 4. The run

- **Defaults:** from scratch, batch 512, LR 5e-4, FP32 with TF32 tensor cores (~16 GiB of GPU
  memory), up to 150 epochs of ~500 steps each, with a validation and a PAI score every 500
  steps (about once per epoch). PAI usually ends the run itself after adding 2 rounds of dendrites.
- **Where things go:** W&B team `IdlF26Group40`, project `ViT-perforated`; checkpoints and PAI's
  graphs in `~/runs/<run name>/` (or `$LOG_ROOT`).
- **Variants** (environment variables; on Slurm pass them with `sbatch --export=ALL,NAME=value`):

  | Variable | Effect |
  |---|---|
  | `CONTROL=1` | Same wrapping and schedule, no dendrites (the baseline to compare against) |
  | `PAI_INIT=<ckpt>` | Fine-tune a trained baseline instead of training from scratch |
  | `AMP=1` | fp16 mixed precision instead of FP32 (~9 GiB instead of ~16 GiB) |
  | `BATCH_SIZE=384` | If the GPU runs out of memory |
  | `WORKERS=<n>` | Data-loader workers (default: CPUs − 1, at most 8) |
  | `EPOCHS`, `LR`, `WARMUP`, `MAX_DENDRITES`, `LOG_STEPS`, `TAG` | See the script header |

  Arguments after the script name go to `main.py`, e.g. `--seed 0` or `--max-steps 50`.

- **Continuing a stopped run** (Slurm time limit, reboot): pass the run name, which is the folder
  name under `~/runs`. PAI can only resume from a checkpoint saved in a neuron phase; the script
  stops with an error otherwise.
  ```bash
  CONTINUE_RUN=<run name> ./scripts/train_vit_tiny_pai_last6_ln_mlp.sh                  # personal GPU
  sbatch --export=ALL,CONTINUE_RUN=<run name> ... scripts/train_vit_tiny_pai_last6_ln_mlp.sbatch  # Slurm
  ```

## Reproducibility

The 20% training subset is always the same (`--subset-seed 42`, independent of everything else),
and validation uses all 50k images with a fixed crop. Weight init, shuffle order and
augmentations are random unless you pass `--seed`; even then they only repeat with the same
`WORKERS` and `BATCH_SIZE`, and GPU arithmetic is not bit-for-bit deterministic.

## Package versions

`perforatedai==4.0.2` and `perforatedbp==4.0.1` (pinned in `pyproject.toml` / `uv.lock`). The
candidate loss is perforatedbp's default, `ImprovedCovarianceDendriteLoss`; its scores were stuck
at 0 in 4.0.0. `pai_setup.py` sets `pai_improvement_threshold_raw` to 0, which these versions need.
