#!/bin/bash
# Table 2 of differing_slopes.tex §5.2: bootstrap coverage of the four-role estimator in the
# nonlinear displayed design (paper_nonlinear), 999 full re-estimation draws per data set,
# 500 data sets at n = 4,000 and 16,000 and 200 at n = 64,000 (as in the linear-design run
# runs/ds_four_role_coverage_999). Each array task runs one shard of replications; seeds depend
# only on (seed, n, rep), so the merged shards equal one run (checked 2026-11-15).
#
# Submit from the repository root on Great Lakes (see experiments/cluster/README.md):
#   OUT=experiments/runs/ds_four_role_coverage_nl_999; mkdir -p $OUT/logs
#   sbatch --output=$OUT/logs/%x_%A_%a.out --export=ALL,OUT=$OUT \
#       experiments/cluster/ds_section5_coverage_array.sbatch
#   sbatch --dependency=afterok:<array job id> --output=$OUT/logs/%x_%j.out \
#       --export=ALL,OUT=$OUT experiments/cluster/ds_section5_coverage_merge.sbatch
#
# Sizes come from the test job 66189320 (debug partition, 2026-11-15): 0.095, 0.245 and 0.82 s
# per fit at n = 4k, 16k, 64k and 190 MB peak memory for two workers. Each task below runs
# 20-27 minutes on 4 CPUs; the limit is about 1.7 times that. 93 CPU-hours in all, about $1.40
# at the 2026-11-15 rate (billing 2504.5 per CPU-minute; dollars = billing-minutes / 1e7).
#SBATCH --job-name=ds5-coverage
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:45:00
#SBATCH --array=0-59
#SBATCH --requeue

set -euo pipefail

# Shard table: n, first replication, one past the last. Shards are sized so that each task
# takes about the same time; the number of entries must match --array.
SHARDS=()
add_cell() {  # n, replications, replications per shard
    local n=$1 reps=$2 size=$3 a b
    for ((a = 0; a < reps; a += size)); do
        b=$((a + size < reps ? a + size : reps))
        SHARDS+=("$n $a $b")
    done
}
add_cell 4000 500 50    # 10 shards, 13 rounds of 4 workers
add_cell 16000 500 20   # 25 shards, 5 rounds
add_cell 64000 200 8    # 25 shards, 2 rounds

if (( SLURM_ARRAY_TASK_ID >= ${#SHARDS[@]} )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} has no shard (only ${#SHARDS[@]})" >&2
    exit 1
fi
read -r N A B <<< "${SHARDS[$SLURM_ARRAY_TASK_ID]}"

OUT="${OUT:?set OUT to the run directory}"
SHARD_OUT="${OUT}/shards/n${N}_r${A}-${B}"
mkdir -p "${SHARD_OUT}"

# One numerical-library thread per Python worker; the process pool supplies the parallelism.
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

srun --cpu-bind=cores "$PY" -m experiments.scripts.ds_four_role_coverage \
    --design displayed --n "$N" --rep-start "$A" --reps "$B" --boot 999 \
    --workers "${SLURM_CPUS_PER_TASK}" --out "${SHARD_OUT}"
