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

## 1. Build the image

```bash
git clone <repo url> ViT-perforated && cd ViT-perforated
git switch l40-pai-resume
docker build -f docker/Dockerfile -t vit-perforated-l40 .
```

The build downloads about 5 GB. If you were given the image as a file instead, load it with
`docker load -i vit-perforated-l40.tar.gz`.

## 2. Add your keys

Skip this step if you were given the `vit-perforated-l40:keys` image, which has the keys built
in. `docker/run_l40.sh` uses that image automatically when the machine has it. Otherwise:

```bash
cp .env.example .env && chmod 600 .env
nano .env    # fill in WANDB_API_TOKEN, PAIEMAIL and PAITOKEN
```

`.env` is mounted read-only when the container runs. It is never copied into the plain image. A
`.env` also overrides the keys built into the `:keys` image.

To make a `:keys` image for someone else, build the plain image and run `docker/bake_keys.sh`.
Anyone who has that image can read the keys out of it, so send it only as a file
(`docker save vit-perforated-l40:keys | gzip > vit-perforated-l40-keys.tar.gz`), never push it to a
public registry, and remember that every run started from it is logged under your W&B user.

## 3. Run setup

```bash
docker/run_l40.sh setup
```

This downloads the ImageNet parquet shards (about 22 GB) to `~/vit-data/imagenet_1k_resized_256`,
checks the keys, and runs the GPU smoke test, which includes a 30-step PAI run. Re-running it is
safe because it skips the steps that are already done.

## 4. Train

```bash
docker/run_l40.sh train SCRATCH=1    # fresh run: dendrites on layers 9, 10 and 11
```

The run starts in the background, and the script prints the commands to follow and stop it.
The job first unpacks the shards to `~/vit-data/unpack`, which takes a few minutes, and deletes
them again when it ends. Runs go to `~/vit-data/runs/<run name>` and to W&B.

To resume from a pre-switch checkpoint, put the snapshot folder under `~/vit-data/checkpoints/`.
The folder must keep its `pai/best_model.pt`. Then:

```bash
docker/run_l40.sh train    # default: checkpoints/vit_tiny_pai_scratch_11_20pct_20261111_0433/archive/1111_234822_step61000
docker/run_l40.sh train RESUME=/data/checkpoints/<run>/archive/<snapshot>/checkpoint.pth.tar
```

Paths inside `VAR=value` are container paths: `~/vit-data` on the host is `/data` in the container.

Other settings are passed the same way. The full list is in the header of
`scripts/l40_pai_resume.sbatch`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `ARM` | `9,10,11` | layers with dendrites: a list, one layer (`11`), a range (`6-11`) or `control` |
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

To start a new run from the `pre_switch` checkpoint (the start of the first dendrite phase), on
any machine:

```bash
docker/run_l40.sh train RESUME_WANDB=<run name>            # e.g. other layers: add ARM=11
```

This downloads that checkpoint to `~/vit-data/checkpoints/<run name>/archive/pre_switch/`, resumes
from it with `--pai-resume`, and copies the source run's W&B history up to that step.

## On a Slurm cluster

Most clusters do not allow Docker. Without Docker, run `scripts/setup_machine.sh` on the login
node and submit `scripts/l40_pai_resume.sbatch` with `sbatch`; the header of that script explains
how. To use this image there anyway, convert it with Apptainer:
`apptainer build vit.sif docker-daemon://vit-perforated-l40:latest`.
