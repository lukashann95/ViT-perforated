# Running PAI training on an L40 with Docker

The image holds the code and the Python environments. The dataset, your keys and the training
output stay on the host machine and are mounted into the container.

## What the host needs

- An NVIDIA L40 or L40S. Other GPUs from the A100 to the H100 generation also work. RTX 50-series
  (Blackwell) GPUs do not: the pinned torch 2.6.0 has no kernels for them.
- NVIDIA driver 550 or newer. Check with `nvidia-smi`.
- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Check with `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- About 25 GB of disk for the dataset, plus 40 GB free on a fast local disk for each running job.

## 1. Get the repository

```bash
git clone https://github.com/lukashann95/ViT-perforated.git && cd ViT-perforated
git switch l40-pai-resume
```

You need the repository in both cases below, for `docker/run_l40.sh`. Run all later commands from
this folder.

## 2. Get the image and the keys

Use one of the two ways, depending on what you were sent. To send them to someone, see
[Sharing the image or the keys](#sharing-the-image-or-the-keys).

### Option A: you were sent a `.env` file

Build the image from the Dockerfile. The build downloads about 5 GB:

```bash
docker build -f docker/Dockerfile -t vit-perforated-l40 .
```

Put the `.env` you were sent in the repository folder (next to `.env.example`), and make it
readable only by you:

```bash
cp /path/to/received/.env .env && chmod 600 .env
```

`docker/run_l40.sh` mounts `.env` into the container read-only at every run, and the build never
copies it into the image (`.dockerignore` excludes it). To use your own keys instead, create the
file from the template with `cp .env.example .env`, then fill in `WANDB_API_TOKEN`, `PAIEMAIL`
and `PAITOKEN`.

### Option B: you were sent the image file `vit-perforated-l40-keys.tar.gz`

Load it. The keys are built in, so you need no `.env` and no build:

```bash
docker load -i vit-perforated-l40-keys.tar.gz    # prints: Loaded image: vit-perforated-l40:keys
```

`docker/run_l40.sh` uses the `vit-perforated-l40:keys` image automatically when the machine has
it. A `.env` in the repository folder would override the built-in keys.

### Either way: after code updates

The container runs the copy of the code inside the image, not the files in your checkout. After a
`git pull` with Option A, rebuild the image. With Option B, ask for a new image file.

## 3. Run setup

```bash
docker/run_l40.sh setup
```

This downloads the ImageNet parquet shards (about 22 GB) to `~/vit-data/imagenet_1k_resized_256`,
checks the keys, and runs the GPU smoke test, which includes a 30-step PAI run. Re-running it is
safe because it skips the steps that are already done.

## 4. Train

Every run puts dendrites on layers 9, 10 and 11 unless you set `ARM`. Pick where it starts:

| Start from | Command |
| --- | --- |
| scratch (random initialization) | `docker/run_l40.sh train SCRATCH=1` |
| a run's W&B `pre_switch` checkpoint | `docker/run_l40.sh train RESUME_WANDB=<run name>` |
| a pre-switch snapshot on disk | `docker/run_l40.sh train RESUME=/data/checkpoints/<run>/archive/<snapshot>/checkpoint.pth.tar` |

`train` with none of these settings does not start from scratch. It resumes from the step-61000
snapshot of `vit_tiny_pai_scratch_11_20pct_20261111_0433`, which exists only on Delta, so the job
stops at once unless you have copied that folder to
`~/vit-data/checkpoints/vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/1111_234822_step61000/`.

What the starting points mean:

- **`SCRATCH=1`** trains a new model, with a linear LR warmup over the first 2000 steps
  (`PAI_WARMUP`). PAI switches to a dendrite phase once validation stops improving.
- **`RESUME_WANDB=<run name>`** downloads the `pre_switch` checkpoint of that run (see
  [Checkpoints on W&B](#checkpoints-on-wb)) to `~/vit-data/checkpoints/<run name>/archive/pre_switch/`.
  It then continues the run's neuron phase from there, with this run's layers, and copies the
  source run's W&B history up to that step. Combine it with `ARM` to try other layers from the same
  starting point.
- **`RESUME=<path>`** does the same from a snapshot folder you copied yourself, for example a
  run's `archive/pre_switch/` folder. The folder must hold both `checkpoint.pth.tar` and
  `pai/best_model.pt`. Paths inside `VAR=value` are container paths: `~/vit-data` on the host is
  `/data` in the container.

The run starts in the background, and the script prints the commands to follow and stop it. The
log is also saved to `~/vit-data/runs/docker-<time>.out`. The job first unpacks the shards to
`~/vit-data/unpack`, which takes a few minutes, and deletes them again when it ends. Runs go to
`~/vit-data/runs/<run name>` and to W&B.

To continue a run that was stopped, from its last checkpoint on disk, pass its run name together
with the settings it was started with. For example, for a run started with `SCRATCH=1`:

```bash
docker/run_l40.sh train SCRATCH=1 CONTINUE_RUN=vit_tiny_pai_scratch_9_10_11_ln_mlp_pb_fp32_20pct_<date>
```

The run must have been stopped in a neuron phase. W&B shows the continuation as a new run,
`<run name>_cont<time>`, with the original run's history copied in.

Other settings are passed the same way. The full list is in the header of
`scripts/l40_pai_resume.sbatch`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `SCRATCH` | `0` | `1`: train from random initialization instead of resuming |
| `RESUME` | the Delta step-61000 snapshot | pre-switch checkpoint to resume from (container path) |
| `ARM` | `9,10,11` | layers with dendrites: a list, one layer (`11`), a range (`6-11`) or `control` |
| `EPOCHS` | `350` | maximum epochs, counting those of the run `RESUME` comes from; PAI usually ends the run earlier |
| `PAI_PATIENCE` | `5` | validations without improvement before the LR drops |
| `PAI_N_EPOCHS_TO_SWITCH` | `15` | minimum validations in a neuron phase before PAI may switch |
| `MAX_DENDRITES` | `2` | maximum number of dendrite rounds |
| `WANDB_CKPT` | `1` | upload the last, best and pre-switch checkpoints to W&B (see below); `0` turns it off |
| `RESUME_WANDB` | | resume from that run's W&B pre-switch checkpoint (see below) |
| `CONTINUE_RUN` | | continue a stopped run by its run name; pass its original settings too |

Settings for `docker/run_l40.sh` itself go before the command:

```bash
GPU=1 HOST_DATA=/big/disk/vit-data HOST_UNPACK=/local/nvme/unpack docker/run_l40.sh train SCRATCH=1
```

- `GPU`: which GPU to use (default 0). A PAI run uses exactly one GPU, so run one job per GPU.
- `HOST_DATA`: the data folder on the host (default `~/vit-data`).
- `HOST_UNPACK`: where each job unpacks ImageNet (default `$HOST_DATA/unpack`).
- `ENV_FILE`: the keys file (default: the repo's `.env`).
- `IMAGE`: the image to run (default `vit-perforated-l40:keys` if the machine has it, otherwise
  `vit-perforated-l40`).
- `DOCKER_ARGS`: extra `docker run` options. If downloads hang or fail with "Temporary failure
  resolving", for example on a VPN, try `DOCKER_ARGS="--network host"`.

To get a shell with the same mounts, run `docker/run_l40.sh shell`.

## Checkpoints on W&B

Each run keeps three checkpoints in the W&B model artifact `checkpoint-<run name>`:

| Alias | Checkpoint |
| --- | --- |
| `last` | the latest validation, uploaded at every validation |
| `best` | the validation with the best acc@1 so far |
| `pre_switch` | the last validation before PAI first switched to a dendrite phase |

Each one holds `checkpoint.pth.tar` and `pai/best_model.pt`. Older versions are deleted, so a run
uses about 300 MB on W&B. One version can carry two aliases when they are the same checkpoint.
The checkpoints also stay on disk in `~/vit-data/runs/<run name>/checkpoints/` and
`archive/pre_switch/`. `WANDB_CKPT=0` turns the uploads off.

To download one:

```python
import wandb
path = wandb.Api().artifact("IdlF26Group40/ViT-perforated/checkpoint-<run name>:best").download()
```

To start a new run from a `pre_switch` checkpoint, use `RESUME_WANDB=<run name>` (see
[4. Train](#4-train)).

## Sharing the image or the keys

Whoever sends the keys does this on their own machine, from the repository folder.

- **Option A, send the `.env`:** the file is tiny. Send it over a private channel (for example
  `scp`, or a direct message), never in a git commit.
- **Option B, send the image:** build the plain image, add the keys, and save it to a file of
  about 3.3 GB:

  ```bash
  docker build -f docker/Dockerfile -t vit-perforated-l40 .
  docker/bake_keys.sh
  (umask 077; docker save vit-perforated-l40:keys | gzip -1 > ~/vit-perforated-l40-keys.tar.gz)
  ```

  Send the file directly, for example with `rsync -P ~/vit-perforated-l40-keys.tar.gz
  <user>@<host>:~/`, or with an end-to-end encrypted tool such as `croc`. A cloud drive also
  works if the file is shared with that one person only and deleted afterwards. Never push the
  image to Docker Hub or any other public registry.

Either way, the keys can be read by whoever receives them (`docker run ... cat .env` with the
image), and every W&B run started with them is logged under the W&B user who owns
`WANDB_API_TOKEN`. If the keys leak, revoke the W&B key at https://wandb.ai/authorize and ask
PerforatedAI for a new token.

## On a Slurm cluster

Most clusters do not allow Docker. Without Docker, run `scripts/setup_machine.sh` on the login
node to install the environments, check the keys and download the data. Then submit
`scripts/l40_pai_resume.sbatch` from the repository folder. It takes the same settings as
`docker/run_l40.sh train`, passed with `--export`, and has the same default: without `SCRATCH`,
`RESUME_WANDB` or `RESUME`, it resumes from the Delta snapshot and stops if that is missing.

```bash
cd ViT-perforated
sbatch -p <partition> -A <account> --export=ALL,SCRATCH=1 scripts/l40_pai_resume.sbatch                 # from scratch
sbatch -p <partition> -A <account> --export=ALL,RESUME_WANDB=<run name> scripts/l40_pai_resume.sbatch   # from a W&B pre_switch checkpoint
sbatch -p <partition> -A <account> --export=ALL,ARM=11,SCRATCH=1 scripts/l40_pai_resume.sbatch          # other layers
```

Use your cluster's partition and account; the script's header lists every setting. A job may run
for up to 48 hours. If your partition allows less, add for example `--time=24:00:00` and
continue the run afterwards with `CONTINUE_RUN` (see [4. Train](#4-train)). With Docker there is
no time limit. Without Docker,
paths are host paths: the data is in `<repo parent>/imagenet_1k_resized_256`, runs go to
`<repo parent>/runs`, the job unpacks the data to `$SLURM_TMPDIR` (or `/tmp`), the keys are read
from the repository's `.env`, and the Slurm log is `slurm-vit-pai-l40-<job id>.out` in the folder
you submitted from.

To use the Docker image on such a cluster anyway, convert it with Apptainer:
`apptainer build vit.sif docker-daemon://vit-perforated-l40:latest`.
