#!/bin/bash
#SBATCH --job-name=vit-benchmark
#SBATCH --account=cis260364p
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

# PerforatedAI credentials (PAIEMAIL/PAITOKEN), if this user has them.
if [ -f "$HOME/.pai_env" ]; then source "$HOME/.pai_env"; fi

REPO="/ocean/projects/cis260364p/shared/ViT-perforated"
OUTPUT="/ocean/projects/cis260364p/shared/ViT-perforated/benchmark-${SLURM_JOB_ID}"
# Override duration at submission, e.g. sbatch --export=ALL,STEPS=50 ...
STEPS="${STEPS:-100}"
mkdir -p "$OUTPUT"
cd "$REPO"

# No ImageNet path or copying: --fake-data generates samples on the CPU.
# One AMP run per model, with accumulation 1 (full microbatch of 1024).
# Full synthetic validation and checkpoint saving occur at the final step.
# W&B is omitted to make these smoke tests independent of authentication.
# Total elapsed time includes compilation, validation, and checkpoint writes;
# compare later training Time values (outside parentheses) for step throughput.
printf 'model\tprecision\taccum_freq\tmicrobatch_size\texit_code\telapsed_seconds\n' > "$OUTPUT/summary.tsv"
failed=0
for model in small tiny; do
    if [[ "$model" == small ]]; then
        hidden=384
        heads=6
    else
        hidden=192
        heads=3
    fi
    for ACCUM_FREQ in 1; do
        microbatch_size=$((1024 / ACCUM_FREQ))
        name="vit_${model}_amp_accum${ACCUM_FREQ}"
        echo "Starting $name: $STEPS steps, accumulation $ACCUM_FREQ"
        started=$SECONDS
        status=0
        uv run python -u main.py \
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
            --workers 5 \
            --logs "$OUTPUT" \
            --name "$name" \
            --amp > "$OUTPUT/$name.log" 2>&1 || status=$?
        elapsed=$((SECONDS - started))
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$model" amp "$ACCUM_FREQ" "$microbatch_size" "$status" "$elapsed" >> "$OUTPUT/summary.tsv"
        echo "Finished $name: exit=$status, elapsed=${elapsed}s; log: $OUTPUT/$name.log"
        if (( status != 0 )); then
            failed=1
        fi
    done
done
cat "$OUTPUT/summary.tsv"
exit "$failed"
