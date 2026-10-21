#!/bin/bash
#SBATCH --job-name=vit-benchmark
#SBATCH --account=TODO
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

module load anaconda3
conda deactivate
conda activate /ocean/projects/cis250019p/mzhang23/TA/envs/IDLF26
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1

REPO="${REPO:-$HOME/ViT-perforated}"
OUTPUT="${OUTPUT:-$PROJECT/vit_perforated/benchmark-${SLURM_JOB_ID}}"
# Override at submission, e.g. sbatch --export=ALL,STEPS=50,ACCUM_FREQ=32 ...
STEPS="${STEPS:-100}"
ACCUM_FREQ="${ACCUM_FREQ:-16}"
mkdir -p "$OUTPUT"
cd "$REPO"

# No ImageNet path or copying: --fake-data generates samples on the CPU.
# Same total batch and microbatch for all runs (1024 / 16 = 64 by default).
# Full synthetic validation and checkpoint saving occur at the final step.
# W&B is omitted to make these smoke tests independent of authentication.
# Total elapsed time includes compilation, validation, and checkpoint writes;
# compare later training Time values (outside parentheses) for step throughput.
printf 'model\tprecision\texit_code\telapsed_seconds\n' > "$OUTPUT/summary.tsv"
failed=0
for model in small tiny; do
    if [[ "$model" == small ]]; then
        hidden=384
        heads=6
    else
        hidden=192
        heads=3
    fi
    for precision in fp32 amp; do
        amp_args=()
        if [[ "$precision" == amp ]]; then
            amp_args=(--amp)
        fi
        name="vit_${model}_${precision}"
        echo "Starting $name: $STEPS steps, accumulation $ACCUM_FREQ"
        started=$SECONDS
        status=0
        python -u main.py \
            --fake-data \
            --hidden-dim "$hidden" \
            --num-layers 12 \
            --num-heads "$heads" \
            --patch-size 16 \
            --batch-size 1024 \
            --accum-freq "$ACCUM_FREQ" \
            --mlp-head \
            --torchvision-inception-crop \
            --epochs 90 \
            --max-steps "$STEPS" \
            --print-freq 10 \
            --log-steps "$STEPS" \
            --workers 8 \
            --logs "$OUTPUT" \
            --name "$name" \
            "${amp_args[@]}" > "$OUTPUT/$name.log" 2>&1 || status=$?
        elapsed=$((SECONDS - started))
        printf '%s\t%s\t%s\t%s\n' "$model" "$precision" "$status" "$elapsed" >> "$OUTPUT/summary.tsv"
        echo "Finished $name: exit=$status, elapsed=${elapsed}s; log: $OUTPUT/$name.log"
        if (( status != 0 )); then
            failed=1
        fi
    done
done
cat "$OUTPUT/summary.tsv"
exit "$failed"
