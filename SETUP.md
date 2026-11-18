# Setting up a GPU machine

`scripts/setup_machine.sh` prepares a machine to run this repo's experiments. It starts no
training run. Supported targets:

| Machine | GPU | Notes |
|---|---|---|
| Personal box | RTX 3090 (24 GB) / RTX 4090 (24 GB) | No Slurm |
| Slurm cluster | L40S (48 GB) | Run setup on the login node |
| NCSA Delta | A40 (48 GB) / A100 | Run setup on the login node; the dataset is already there |

All of them work with the locked PyTorch 2.6 / CUDA 12.4 build, which needs NVIDIA driver ≥ 550.

What setup does (each step is skipped if it's already done, so you can re-run it):

1. Installs `uv`.
2. Creates the training venv (`.venv`) from `uv.lock`, plus a small `.venv-data` with pyarrow
   for unpacking the dataset.
3. Checks that `.env` holds the W&B and PerforatedAI keys.
4. Downloads ImageNet-1k (`evanarlian/imagenet_1k_resized_256`, train + val, ~22 GB of parquet
   shards). **The download is not unpacked.** Each job unpacks it to node-local disk (see
   [Using the data in a job](#using-the-data-in-a-job)).
5. Runs a smoke test if a GPU is visible: GPU and torch kernels, credentials, perforation of
   layer 11, and a 30-step PAI run on fake data.

Settings are environment variables, listed in the header of `scripts/setup_machine.sh`
(`VENV`, `DATA_VENV`, `DATA_DIR`, `PARQUET_DIR`, `DOWNLOAD_DATA=0`, `DATA_INCLUDE`, `SKIP_SMOKE=1`).

## 1. Get the code

```bash
git clone git@github.com:lukashann95/ViT-perforated.git
cd ViT-perforated
```

## 2. Credentials: the `.env` file

The runs need a W&B API key and a PerforatedAI license. They go in **`.env`** in the repo root.
Git ignores `.env`, so it is never pushed, and `git clone` doesn't bring it either. Every machine
needs its own copy. `.env.example` is the committed template. **Never put keys in it.**

```bash
WANDB_API_TOKEN=<your W&B API key, from https://wandb.ai/authorize>
WANDB_ENTITY=IdlF26Group40
WANDB_PROJECT=ViT-perforated
PAIEMAIL=<PerforatedAI license email>
PAITOKEN=<PerforatedAI license token>
```

Write it as one `KEY=value` per line, with no quotes and no spaces around `=`. Either create it
on the machine (`cp .env.example .env && chmod 600 .env`, then edit it), or copy yours over with
`scp .env <user>@<host>:<path>/ViT-perforated/.env`. If `.env` is missing, setup creates it,
stops at step 3 and lists the missing keys. Fill them in and re-run.

## 3. Run setup

### RTX 3090 / RTX 4090 (personal machine)

You need ~25 GB of free disk and internet access.

```bash
./scripts/setup_machine.sh            # ~20 min, mostly the download
```

The shards go to `~/data/imagenet_1k_resized_256/data`; set `DATA_DIR=...` to put them
somewhere else. Setup ends with `SMOKE TEST PASSED` and `PAI training run ok`.

### L40S (Slurm cluster)

Login nodes usually have internet access but no GPU. Run setup there; it skips the smoke test.

```bash
DATA_DIR=/path/with/space ./scripts/setup_machine.sh
# optional: the GPU smoke test, inside a job (finished steps are skipped)
srun -p <partition> -A <account> --gres=gpu:l40s:1 --cpus-per-task=8 --time=1:00:00 \
    --pty env DATA_DIR=/path/with/space ./scripts/setup_machine.sh
```

To find your partition and account, run `sinfo -s` and
`sacctmgr show assoc user=$USER format=account,partition`. The GPU name for `--gres` is listed
by `sinfo -o "%P %G"`.

### NCSA Delta (A40)

Delta already has the parquet shards in `/projects/bjto/shared`. `PARQUET_DIR` makes setup
check them instead of downloading them again. It only reads that folder. Keep the repo under the
project directory. Run setup on the login node:

```bash
cd /projects/bjto/$USER
git clone git@github.com:lukashann95/ViT-perforated.git && cd ViT-perforated
# put your .env here (step 2), then:
PARQUET_DIR=/projects/bjto/shared/imagenet_1k_resized_256/data \
DATA_VENV=/projects/bjto/$USER/.venv-data \
    nice -n 19 ./scripts/setup_machine.sh
```

The script ends with `No GPU visible ...`, which is expected on a login node. With
`DATA_VENV=/projects/bjto/$USER/.venv-data`, the venv sits where `scripts/delta_pai_resume.sbatch`
expects it. The optional GPU smoke test:
`srun -A bjto-delta-gpu -p gpuA40x4 --gpus-per-node=1 --time=1:00:00 --pty env PARQUET_DIR=... ./scripts/setup_machine.sh`.

## Using the data in a job

Reading 1.3M small files from shared or network storage is slow, so a job unpacks the shards to
node-local disk at its start. This needs ~25 GB free and takes a few minutes.
`scripts/delta_pai_resume.sbatch` does it this way:

```bash
LOCAL=/tmp/$USER/imagenet_$SLURM_JOB_ID          # or another local scratch dir
mkdir -p "$LOCAL" && trap 'rm -rf "$LOCAL"' EXIT
.venv-data/bin/python scripts/parquet_to_imagefolder.py <parquet dir> "$LOCAL" --workers "$SLURM_CPUS_PER_TASK"
.venv/bin/python -u main.py "$LOCAL" ...
```

`<parquet dir>` is the path that setup prints at the end as `parquet shards:`. On a personal
3090/4090, you can unpack once to a local folder and reuse it, because the disk is already local.
