#!/bin/bash
# Full-sample reuse against the four-role estimator in the Section 5 design (review item m19;
# author, 2026-11-19: "Do both, with as many draws as you think would be good for the paper",
# "send to cluster if necessary"). Script: experiments/scripts/ds_fullsample_reuse.py. Table 2's
# data sets at n = 4k, 16k, 64k, 500 per n, 999 draws: 88 tasks x 4 CPUs, about 110
# CPU-hours (about $1.70). Submit from the repository root on Great Lakes:
#   OUT=experiments/runs/ds_fullsample_reuse_999; mkdir -p $OUT/logs
#   sbatch --array=0-87 --output=$OUT/logs/%x_%A_%a.out --export=ALL,OUT=$OUT \
#       experiments/cluster/ds_fullsample_reuse_array.sbatch
#   sbatch --dependency=afterok:<array id> --output=$OUT/logs/%x_%j.out --export=ALL,OUT=$OUT \
#       experiments/cluster/ds_fullsample_reuse_merge.sbatch
#SBATCH --job-name=ds-fullsample
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=01:00:00
#SBATCH --requeue

set -euo pipefail
SHARDS=()
add_cell() {  # n, replications, replications per shard
    local n=$1 reps=$2 size=$3 a b
    for ((a = 0; a < reps; a += size)); do
        b=$((a + size < reps ? a + size : reps))
        SHARDS+=("$n $a $b")
    done
}
add_cell 4000 500 100     # 5 shards, 25 rounds of 4 workers
add_cell 16000 500 25     # 20 shards
add_cell 64000 500 8      # 63 shards, 2 rounds
if (( SLURM_ARRAY_TASK_ID >= ${#SHARDS[@]} )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} has no shard (only ${#SHARDS[@]})" >&2; exit 1
fi
read -r N A B <<< "${SHARDS[$SLURM_ARRAY_TASK_ID]}"
OUT="${OUT:?set OUT to the run directory}"
SHARD_OUT="${OUT}/shards/full_n${N}_r${A}-${B}"
mkdir -p "${SHARD_OUT}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"
{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
    echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    echo "n=${N} reps=${A}..$((B - 1)) cpus=${SLURM_CPUS_PER_TASK}"
    echo "commit=$(git rev-parse HEAD) python=$("$PY" --version 2>&1)"
    git status --short --untracked-files=no
} > "${SHARD_OUT}/provenance.txt"
srun --cpu-bind=cores "$PY" -m experiments.scripts.ds_fullsample_reuse \
    --n "$N" --rep-start "$A" --reps "$B" --boot 999 \
    --workers "${SLURM_CPUS_PER_TASK}" --out "${SHARD_OUT}"
