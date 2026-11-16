#!/bin/bash
# Table 2 of differing_slopes.tex §5.2 at n = 64,000: 300 more data sets (replications
# 200-499), so the row has 500 like the others (weekly scope review, 2026-11-16). Same
# design, estimator and 999 bootstrap draws as ds_section5_coverage_array.sbatch; seeds
# depend only on (seed, n, rep), so replications 0-199 are unchanged and the new ones are
# exactly what one 500-replication run would give.
#
# The original run directory (ds_four_role_coverage_nl_999, 200 data sets at 64k, the one
# the paper used until now) is left as it is. Its shards are copied into a new directory,
# the new shards go next to them, and the merge job summarizes all of them there.
#
# Submit from the repository root on Great Lakes:
#   OLD=experiments/runs/ds_four_role_coverage_nl_999
#   OUT=experiments/runs/ds_four_role_coverage_nl_999_n64k500
#   mkdir -p $OUT/logs && cp -rp $OLD/shards $OUT/
#   sbatch --output=$OUT/logs/%x_%A_%a.out --export=ALL,OUT=$OUT \
#       experiments/cluster/ds_section5_64k_extend_array.sbatch
#   sbatch --dependency=afterok:<array job id> --output=$OUT/logs/%x_%j.out \
#       --export=ALL,OUT=$OUT,R64=500 experiments/cluster/ds_section5_coverage_merge.sbatch
#
# Sizes from the 2026-11-15 run (array 66189342): 796-1637 s per replication at 64k across
# standard nodes, median about 1014 s, under 1 GB per worker. One round per task (4
# replications on 4 CPUs): about 17 minutes, 27 on the slowest node seen; the limit is 45
# (README lesson: about 2.5 times the median). About 85 CPU-hours, about $1.30.
#SBATCH --job-name=ds5-64k-extend
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:45:00
#SBATCH --array=0-74
#SBATCH --requeue

set -euo pipefail

N=64000
FIRST=200            # first new replication
PER=4                # replications per task: one round on 4 CPUs
A=$((FIRST + PER * SLURM_ARRAY_TASK_ID))
B=$((A + PER))
if (( B > 500 )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} is past replication 499" >&2
    exit 1
fi

OUT="${OUT:?set OUT to the run directory}"
SHARD_OUT="${OUT}/shards/n${N}_r${A}-${B}"
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

srun --cpu-bind=cores "$PY" -m experiments.scripts.ds_four_role_coverage \
    --design displayed --n "$N" --rep-start "$A" --reps "$B" --boot 999 \
    --workers "${SLURM_CPUS_PER_TASK}" --out "${SHARD_OUT}"
