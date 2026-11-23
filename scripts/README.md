# Running on riddle, the 3090 PC, Delta and the L40S

Set up a machine once with `setup_machine.sh` (see the repo's `README.md`). After that you start every run
through one of two launchers. They run unchanged on every machine, because `machine_env.sh` handles what
differs between them.

| Script | What it does |
|---|---|
| `run_vit_tiny_baseline.sh` | plain ViT-Tiny/16 (no PAI), 90-epoch cosine, 20% ImageNet subset |
| `pai_p_lr_ablation.sh` | dendrite-phase LR ablation: perforate block 11 (`TARGET`) of a plain checkpoint, train only the first dendrite phase at five LR scales |
| `machine_env.sh` | sourced by both launchers: dataset, workers, Python, credentials, W&B project, GPU |
| `delta_pai_resume.sbatch` | older Delta-only launcher for plateau-schedule PAI runs (and continuing them after a timeout) |
| `l40_pai_resume.sbatch` | Vamsi's L40 version of it (any Slurm cluster; setup in its header) |
| `l40_pai_p_lr_ablation.sbatch` | both LR sweeps (`ln_mlp`, `ln_fc1`) from the 1024 baseline as one Slurm array job on an L40 / L40S |
| `setup_machine.sh`, `smoke_test.py` | one-time machine setup and its GPU check |
| `parquet_to_imagefolder.py` | unpacks the parquet shards into `train/<wnid>/`, `val/<wnid>/` |
| `export_plain_checkpoint.py` | turns a PAI checkpoint into a plain ViT checkpoint |

Both launchers take their settings from environment variables (listed in each script's header).
`machine_env.sh` only fills in what you leave unset:

| Variable | Default |
|---|---|
| `DATA` (`IMAGENET` for the baseline) | `~/data/imagenet` or `<repo parent>/imagenet` if one exists; otherwise (and always on Delta) the parquet shards are unpacked per job, see below |
| `PARQUET_DIR` | the shards to unpack: Delta's shared copy, elsewhere `<repo parent>/imagenet_1k_resized_256/data` (where `setup_machine.sh` downloads them) |
| `LOGS` (ablation) | `~/runs`; Delta: `/projects/bjto/$USER/runs`; L40 sbatch: `<repo parent>/runs` |
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

### L40S

A Slurm cluster run by a friend of Vamsi's. The L40S has 48 GB, so batch 1024 fits in one pass
(`ACCUM=1`). Set the machine up once with `scripts/setup_machine.sh` on the login node (venvs, keys in
`.env`, parquet shards in `<repo parent>/imagenet_1k_resized_256/data`), then submit
`l40_pai_p_lr_ablation.sbatch` (the sweeps) or `l40_pai_resume.sbatch` (plateau runs) from the repo root
with the cluster's partition and account (`-p`, `-A`). `sbatch -p <partition> -A <account>
scripts/l40_pai_p_lr_ablation.sbatch` alone runs both batch-1024 sweeps, one array task per target. Each job unpacks the shards to `$SLURM_TMPDIR`
(or `/tmp`); runs go to `<repo parent>/runs`, the Slurm log to the directory you submitted from.

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
It is either in the run's `logs/<run>/checkpoints/` on the machine that trained it (`CKPT=<path>`), or on
W&B: `CKPT_WANDB=<run>` downloads `checkpoint-<run>:last` to `$LOGS/ckpts/<run>/` (once; later sweeps reuse it).

`TARGET` picks what gets dendrites in block 11: `ln_mlp` (default: `ln_2` with the whole MLP) or `ln_fc1`
(`ln_2` with the first Linear only); also `fc1`, `fc2`, `mlp`.

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
GPU=1 WORKERS=8 CKPT=<checkpoint> PEAK_LR=5e-4 BATCH=256 TARGET=ln_fc1 ./scripts/pai_p_lr_ablation.sh
# Delta
sbatch --export=ALL,CKPT=<checkpoint>,PEAK_LR=1e-3,BATCH=1024 scripts/pai_p_lr_ablation.sh
# L40S: both targets from the 1024 baseline (array tasks 0 = ln_mlp, 1 = ln_fc1)
sbatch -p <partition> -A <account> scripts/l40_pai_p_lr_ablation.sbatch
```

Runs are named `vit_tiny_pai_plr_l11_<target>_b<batch>_acc<accum>_lr<peak LR>_k<k>_<time>` (runs before
11-22 lack the target and LR). Judge them by PB scores, not by validation accuracy: W&B logs the best score
so far as `pai/pb_best/<module>` (e.g. `layer11.mlp` for `ln_mlp`, `layer11.mlp.0` for `ln_fc1`) and keeps
PAI's csv files under `pai/`; the plots are in `<run>/pai_images`. Per Rorry, PB scores under 0.01 are noise.
