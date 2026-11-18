#!/bin/bash
# Simulations for the predicted referee concerns (author, 2026-11-18: "do all five").
# Script: experiments/scripts/ds_review_ablations.py (designs in its docstring). Eleven
# designs, 500 data sets per cell, 999 full re-estimation draws each (cluster_g31 also runs
# the group bootstrap on every data set). Seeds depend only on (seed, n, rep), so the merged
# shards equal one run.
#
# Two shard sets, submitted as two arrays because their run times differ:
#   SET=short  every design at n = 16k and 64k: 869 tasks (4 CPUs)
#   SET=long   every design at n = 256k: 693 tasks of one round (8 data sets on 8 CPUs)
# Submit from the repository root on Great Lakes (see the README run log):
#   OUT=experiments/runs/ds_review_ablations_999; mkdir -p $OUT/logs
#   sbatch --array=0-692 --cpus-per-task=8 --mem-per-cpu=2G --time=03:00:00 \
#       --output=$OUT/logs/%x_%A_%a.out --export=ALL,OUT=$OUT,SET=long \
#       experiments/cluster/ds_review_ablations_array.sbatch
#   sbatch --array=0-868 --output=$OUT/logs/%x_%A_%a.out --export=ALL,OUT=$OUT,SET=short \
#       experiments/cluster/ds_review_ablations_array.sbatch
#   sbatch --dependency=afterok:<long id>:<short id> --output=$OUT/logs/%x_%j.out \
#       --export=ALL,OUT=$OUT experiments/cluster/ds_review_ablations_merge.sbatch
#SBATCH --job-name=ds-review
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:30:00
#SBATCH --requeue

set -euo pipefail

DESIGNS="cluster_g31 interact_k10 interact_k20 grid_h27 flat_l50 flat_l25 skew hetero_a148 hetero_b148 hetero_ca hetero_cb"
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
            add_cell "$d" 16000 500 32     # 16 shards, 8 rounds of 4 workers
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

srun --cpu-bind=cores "$PY" -m experiments.scripts.ds_review_ablations \
    --design "$DESIGN" --n "$N" --rep-start "$A" --reps "$B" --boot 999 \
    --workers "${SLURM_CPUS_PER_TASK}" --out "${SHARD_OUT}"
