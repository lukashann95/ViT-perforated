# Running on riddle, the 3090 PC and Delta

Set up a machine once with `setup_machine.sh` (see the repo's `README.md`). After that you start every run
through one of two launchers. They run unchanged on every machine, because `machine_env.sh` handles what
differs between them.

| Script | What it does |
|---|---|
| `run_vit_tiny_baseline.sh` | plain ViT-Tiny/16 (no PAI), 90-epoch cosine, 20% ImageNet subset |
| `pai_p_lr_ablation.sh` | dendrite-phase LR ablation: perforate block 11 of a plain checkpoint, train only the first dendrite phase at five LR scales |
| `machine_env.sh` | sourced by both launchers: dataset, workers, Python, credentials, W&B project, GPU |
| `delta_pai_resume.sbatch` | older Delta-only launcher for plateau-schedule PAI runs (and continuing them after a timeout) |
| `setup_machine.sh`, `smoke_test.py` | one-time machine setup and its GPU check |
| `parquet_to_imagefolder.py` | unpacks the parquet shards into `train/<wnid>/`, `val/<wnid>/` |
| `export_plain_checkpoint.py` | turns a PAI checkpoint into a plain ViT checkpoint |

Both launchers take their settings from environment variables (listed in each script's header).
`machine_env.sh` only fills in what you leave unset:

| Variable | Default |
|---|---|
| `DATA` (`IMAGENET` for the baseline) | `~/data/imagenet`; Delta: unpacked per job, see below |
| `GPU` | `0`; under Slurm the GPU the job was given |
| `WORKERS` | CPU cores - 1; under Slurm, CPUs - 2 |
| `WANDB_ENTITY` / `WANDB_PROJECT` | `IdlF26Group40` / `ViT-perforated` |
| `PYTHON` | `<repo>/.venv/bin/python` (after a `uv.lock` change, run `uv sync --frozen`) |

Credentials come from the repo's `.env` (`WANDB_API_TOKEN`, and `PAIEMAIL`/`PAITOKEN` unless they are in
`~/.pai_env`). Variables you set on the command line override both files.

## Machines

| | riddle | 3090 PC | Delta |
|---|---|---|---|
| Who | shared CMU box (other users too) | Anshu only | Lukas (allocation `bjto`) |
| GPUs | 2× RTX 4090 (24 GB) | 2× RTX 3090 (24 GB) | A40 (48 GB) per job |
| RAM / CPU | 61 GB, 16 cores (shared!) | 20 threads | 64 GB, 16 CPUs per job |
| Repo | `~/ViT-perforated` (Lukas), `/usr1/home/vamsig/IDL/IDL-project/ViT-perforated` (Vamsi) | `/home/udaygirish/Anshu/ViT-perforated` | `/projects/bjto/lhann/ViT-perforated` |
| Dataset | `~/data/imagenet` (Lukas), `/usr1/home/vamsig/IDL/IDL-project/imagenet` (Vamsi) | `/home/udaygirish/data/imagenet` (= the default) | `/projects/bjto/shared/imagenet_1k_resized_256/data` (parquet, unpacked by each job) |
| Logs | baseline: `<repo>/logs`; ablation: `~/runs` | same | baseline: `<repo>/logs`; ablation and Slurm output: `/projects/bjto/lhann/runs` |
| Start a run | `tmux`, then `./scripts/...` | `tmux`, then `./scripts/...` | `sbatch scripts/...` from the repo root |

Each user's home on riddle is private: copy checkpoints through W&B or ask the owner.

### riddle

- **Always start runs inside `tmux`** (or a queue). A run started straight from an SSH shell dies when the
  connection drops; that killed the first 1024 baseline (11-21).
- **RAM is the limit, not the GPU.** Each data worker holds about 1 GB, and validation workers join at every
  eval, so `WORKERS=15` costs 25–30 GB. Other users' jobs can grab 20+ GB within minutes, and when free
  memory runs out the system kills whole sessions (second 1024 baseline, 11-21 07:40 UTC). Use `WORKERS=8`.
  Check `free -g` and `nvidia-smi` before starting.
- Lukas queues runs per GPU with task-spooler (`~/.local/bin/forgeq-gpu 0|1`, shared with his other project)
  and wraps long jobs in `~/runs/mem_guard.sh <min GB> <command>`, which stops the job cleanly when free
  RAM drops below the limit. Both live in his home; to run your own, copy them.
- PAI refuses to start when more than one GPU is visible: set `GPU`.

```bash
cd ~/ViT-perforated
tmux new -s baseline
GPU=1 WORKERS=8 ./scripts/run_vit_tiny_baseline.sh
```

### 3090 PC

- Two jobs at once (one per GPU) share 20 threads: use `WORKERS=9` and `WORKERS=8`.
- Both jobs can run from the same checkout, but don't switch branches or pull while either runs: spawned
  data workers re-import `main.py` from disk.

```bash
cd /home/udaygirish/Anshu/ViT-perforated
tmux new -s baseline
GPU=0 WORKERS=9 ./scripts/run_vit_tiny_baseline.sh
```

### Delta

- Log in with Kerberos + Duo. Run `uv` with
  `UV_CACHE_DIR=/projects/bjto/lhann/.uv-cache UV_PYTHON_INSTALL_DIR=/projects/bjto/lhann/.uv-python`
  (everything stays under `/projects`, not `$HOME`).
- Submit from the repo root: `sbatch` runs a copy of the script, so the launchers find the repo through the
  directory you submitted from. Pass settings with `--export=ALL,NAME=value,...`.
- The `#SBATCH` lines at the top of each launcher ask for one A40 (`gpuA40x4`, account `bjto-delta-gpu`),
  16 CPUs, 64 GB and 24 h. Override any of them on the command line, e.g. `sbatch --time=12:00:00 ...`.
  Shorter jobs start sooner (backfill). A busy A40 queue can mean a day's wait.
- Each job unpacks the shared parquet shards to node-local `/tmp/$USER/imagenet_<job id>` (~30 GB, a few
  minutes) and deletes them at the end. One ablation job unpacks once for all five scales.
- PAI credentials: `/projects/bjto/lhann/.pai_env`. Slurm output: `/projects/bjto/lhann/runs/slurm-<job id>.out`.
- A40s have 48 GB, so batch 1024 fits in one pass: add `ACCUM=1`.

```bash
cd /projects/bjto/lhann/ViT-perforated
sbatch --export=ALL,ACCUM=1 scripts/run_vit_tiny_baseline.sh
squeue -u $USER; tail -f /projects/bjto/lhann/runs/slurm-<job id>.out
```

### L40S (babel)

Vamsi's L40S runs use a Docker image and their own launcher on the unmerged `l40-pai-resume` branch
(`docker/README.md`, `scripts/l40_pai_resume.sbatch`). They don't use these launchers yet.

## Recipes

### Plain 90-epoch baselines

Three setups. All use `--wd = 1e-4 × batch/1024`, which the script computes:

```bash
./scripts/run_vit_tiny_baseline.sh                        # batch 1024, LR 1e-3 (4×256 accumulation by default)
BATCH=256 SCALING=linear ./scripts/run_vit_tiny_baseline.sh   # LR 2.5e-4
BATCH=256 SCALING=sqrt ./scripts/run_vit_tiny_baseline.sh     # LR 5e-4
```

Checkpoints are saved at every evaluation (every 2,500 steps, plus epochs 30 and 60) to
`<repo>/logs/<run>/checkpoints/checkpoint.pth.tar`. They are also uploaded to W&B as artifact
`checkpoint-<run>` (aliases `last` and `best`; `WANDB_CKPT=0` turns that off).

**Resume after a crash** with the same name. This continues the same W&B run:
```bash
RUN_NAME=<run> ./scripts/run_vit_tiny_baseline.sh --resume <repo>/logs/<run>/checkpoints/checkpoint.pth.tar
```

### Dendrite-phase LR ablation

You need a finished plain checkpoint. Use `checkpoint.pth.tar` (the final step), not `model_best.pth.tar`.
It is either in the run's `logs/<run>/checkpoints/` on the machine that trained it, or on W&B:

```bash
WANDB_API_KEY=$(grep ^WANDB_API_TOKEN= .env | cut -d= -f2) \
  .venv/bin/wandb artifact get IdlF26Group40/ViT-perforated/checkpoint-<run>:last --root ~/runs/ckpts/<run>
```

One call runs all five scales k ∈ {0.01, 0.1, 1, 10, 100} one after another. Each one takes about 2 h on a
4090 at batch 256. The four setups:

| Setup | Settings |
|---|---|
| batch 256, linear LR | `PEAK_LR=2.5e-4 BATCH=256` |
| batch 256, sqrt LR | `PEAK_LR=5e-4 BATCH=256` |
| batch 1024 | `PEAK_LR=1e-3 BATCH=1024` |
| batch 1024 neurons, 4×256 dendrites | `PEAK_LR=1e-3 BATCH=1024 ACCUM=4` |

```bash
# riddle / 3090 PC
GPU=1 WORKERS=8 CKPT=<checkpoint> PEAK_LR=5e-4 BATCH=256 ./scripts/pai_p_lr_ablation.sh
# Delta
sbatch --export=ALL,CKPT=<checkpoint>,PEAK_LR=1e-3,BATCH=1024 scripts/pai_p_lr_ablation.sh
```

Runs are named `vit_tiny_pai_plr_l11_b<batch>_acc<accum>_k<k>_<time>`. Judge them by the PB-score plots
(`<run>/pai_images`, W&B), not by validation accuracy. Per Rorry, PB scores under 0.01 are noise.
