#!/bin/bash
# Appendix 7 "Non-Gaussian linear predictor": bootstrap coverage in the t5 and mixture designs
# with the nonlinear effect of the Section 5 design, 999 full re-estimation draws per data set,
# 500 data sets per cell (author, 2026-11-16: "Do all sims"). Cells: t5 at n = 4k, 16k, 64k;
# mixture at n = 4k, 16k, 64k, 256k (the mixture criterion is flatter and needs 256k).
# Script: experiments/scripts/ds_nongaussian_coverage.py. Seeds depend only on
# (seed, law, n, rep), so the merged shards equal one run.
#
# Two shard sets, submitted as two arrays because their run times differ:
#   SET=short  t5 and mixture at 4k, 16k, 64k: 330 tasks of about 20 minutes
#   SET=long   mixture at 256k: 125 tasks of one round (4 data sets on 4 CPUs)
# Submit from the repository root on Great Lakes:
#   OUT=experiments/runs/ds_nongaussian_coverage_999; mkdir -p $OUT/logs
#   sbatch --array=0-124 --time=<long limit> --output=$OUT/logs/%x_%A_%a.out \
#       --export=ALL,OUT=$OUT,SET=long experiments/cluster/ds_nongaussian_coverage_array.sbatch
#   sbatch --array=0-329 --output=$OUT/logs/%x_%A_%a.out \
#       --export=ALL,OUT=$OUT,SET=short experiments/cluster/ds_nongaussian_coverage_array.sbatch
#   sbatch --dependency=afterok:<long id>:<short id> --output=$OUT/logs/%x_%j.out \
#       --export=ALL,OUT=$OUT experiments/cluster/ds_nongaussian_coverage_merge.sbatch
# Sizes: see the test job in the README run log.
#SBATCH --job-name=ds-ng-cov
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:50:00
#SBATCH --requeue

set -euo pipefail

SHARDS=()
add_cell() {  # law, n, replications, replications per shard
    local law=$1 n=$2 reps=$3 size=$4 a b
    for ((a = 0; a < reps; a += size)); do
        b=$((a + size < reps ? a + size : reps))
        SHARDS+=("$law $n $a $b")
    done
}
case "${SET:?set SET to short or long}" in
    short)
        for law in t5 mixture; do
            add_cell "$law" 4000 500 64     # 8 shards, 16 rounds of 4 workers
            add_cell "$law" 16000 500 16    # 32 shards, 4 rounds
            add_cell "$law" 64000 500 4     # 125 shards, 1 round
        done ;;
    long)
        add_cell mixture 256000 500 4 ;;   # 125 shards, 1 round
    *) echo "unknown SET=${SET}" >&2; exit 1 ;;
esac

if (( SLURM_ARRAY_TASK_ID >= ${#SHARDS[@]} )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} has no shard (only ${#SHARDS[@]})" >&2
    exit 1
fi
read -r LAW N A B <<< "${SHARDS[$SLURM_ARRAY_TASK_ID]}"

OUT="${OUT:?set OUT to the run directory}"
SHARD_OUT="${OUT}/shards/${LAW}_n${N}_r${A}-${B}"
mkdir -p "${SHARD_OUT}"

export DS_NG_EFFECT=nonlinear
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"

{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
    echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} set=${SET}"
    echo "law=${LAW} n=${N} reps=${A}..$((B - 1)) cpus=${SLURM_CPUS_PER_TASK}"
    echo "commit=$(git rev-parse HEAD) python=$("$PY" --version 2>&1) effect=${DS_NG_EFFECT}"
    git status --short --untracked-files=no
} > "${SHARD_OUT}/provenance.txt"

srun --cpu-bind=cores "$PY" -m experiments.scripts.ds_nongaussian_coverage \
    --law "$LAW" --n "$N" --rep-start "$A" --reps "$B" --boot 999 \
    --workers "${SLURM_CPUS_PER_TASK}" --out "${SHARD_OUT}"
