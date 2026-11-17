#!/bin/bash
# Bootstrap coverage across the first-stage R^2 and the treated share (author, 2026-11-17:
# "do more sims that match this 1) unbalanced treatment setting and 2) different first
# stage R^2 from 0.1 to 0.9"). Script: experiments/scripts/ds_r2_balance_coverage.py.
# Ten designs (R^2 in 0.1, 0.3, 0.5, 0.7, 0.9; half or 11% of units treated), 500 data
# sets per cell, 999 full re-estimation draws each. Seeds depend only on (seed, n, rep),
# so the merged shards equal one run.
#
# Two shard sets, submitted as two arrays because their run times differ:
#   SET=short  every design at n = 4k, 16k, 64k: 850 tasks of about 25 minutes (4 CPUs)
#   SET=long   every design at n = 256k: 630 tasks of one round (8 data sets on 8 CPUs)
# Submit from the repository root on Great Lakes (see the README run log for the limits):
#   OUT=experiments/runs/ds_r2_balance_coverage_999; mkdir -p $OUT/logs
#   sbatch --array=0-629 --cpus-per-task=8 --time=<long limit> --output=$OUT/logs/%x_%A_%a.out \
#       --export=ALL,OUT=$OUT,SET=long experiments/cluster/ds_r2_balance_coverage_array.sbatch
#   sbatch --array=0-849 --output=$OUT/logs/%x_%A_%a.out \
#       --export=ALL,OUT=$OUT,SET=short experiments/cluster/ds_r2_balance_coverage_array.sbatch
#   sbatch --dependency=afterok:<long id>:<short id> --output=$OUT/logs/%x_%j.out \
#       --export=ALL,OUT=$OUT experiments/cluster/ds_r2_balance_coverage_merge.sbatch
#SBATCH --job-name=ds-r2bal
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:15:00
#SBATCH --requeue

set -euo pipefail

DESIGNS="r2_10_bal r2_30_bal r2_50_bal r2_70_bal r2_90_bal r2_10_unbal r2_30_unbal r2_50_unbal r2_70_unbal r2_90_unbal"
SHARDS=()
add_cell() {  # design, n, replications, replications per shard
    local design=$1 n=$2 reps=$3 size=$4 a b
    for ((a = 0; a < reps; a += size)); do
        b=$((a + size < reps ? a + size : reps))
        SHARDS+=("$design $n $a $b")
    done
}
case "${SET:?set SET to short or long}" in
    short)
        for d in $DESIGNS; do
            add_cell "$d" 4000 500 96      # 6 shards, 24 rounds of 4 workers
            add_cell "$d" 16000 500 32     # 16 shards, 8 rounds
            add_cell "$d" 64000 500 8      # 63 shards, 2 rounds
        done ;;
    long)
        for d in $DESIGNS; do
            add_cell "$d" 256000 500 8     # 63 shards, 1 round of 8 workers
        done ;;
    *) echo "unknown SET=${SET}" >&2; exit 1 ;;
esac

if (( SLURM_ARRAY_TASK_ID >= ${#SHARDS[@]} )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} has no shard (only ${#SHARDS[@]})" >&2
    exit 1
fi
read -r DESIGN N A B <<< "${SHARDS[$SLURM_ARRAY_TASK_ID]}"

OUT="${OUT:?set OUT to the run directory}"
SHARD_OUT="${OUT}/shards/${DESIGN}_n${N}_r${A}-${B}"
mkdir -p "${SHARD_OUT}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"

{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
    echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} set=${SET}"
    echo "design=${DESIGN} n=${N} reps=${A}..$((B - 1)) cpus=${SLURM_CPUS_PER_TASK}"
    echo "commit=$(git rev-parse HEAD) python=$("$PY" --version 2>&1)"
    git status --short --untracked-files=no
} > "${SHARD_OUT}/provenance.txt"

srun --cpu-bind=cores "$PY" -m experiments.scripts.ds_r2_balance_coverage \
    --design "$DESIGN" --n "$N" --rep-start "$A" --reps "$B" --boot 999 \
    --workers "${SLURM_CPUS_PER_TASK}" --out "${SHARD_OUT}"
