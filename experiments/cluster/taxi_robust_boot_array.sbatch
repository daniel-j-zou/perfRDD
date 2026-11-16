#!/bin/bash
# Bootstrap intervals (999 draws) for the taxi robustness rows of differing_slopes.tex
# Table taxi-robust that do not have one yet (author, 2026-11-15). All use the main
# specification R2-E1. Already bootstrapped and not rerun: the main row, S1-E1, S2-E1, the
# spline score R3-E1 and the location-scale row. Not bootstrapped: the four other fold
# assignments (their draws would repeat the main row's resamples) and the common slope
# (its estimate sits on the edge of the policy set).
#
# Each array task runs one shard of draws for one row; draw b keeps R2-E1's seeds, so the
# merged shards equal one run (checked 2026-11-15 on S1-E1 draw 1).
#
# Submit from the repository root on Great Lakes (needs the ignored data file
# experiments/datasets/taxi/data/processed/vts_credit.parquet, copied by the author):
#   OUT=experiments/runs/taxi_robust_boot; mkdir -p $OUT/logs
#   sbatch --output=$OUT/logs/%x_%A_%a.out --export=ALL,OUT=$OUT \
#       experiments/cluster/taxi_robust_boot_array.sbatch
#   sbatch --dependency=afterok:<array job id> --output=$OUT/logs/%x_%j.out \
#       --export=ALL,OUT=$OUT experiments/cluster/taxi_robust_boot_merge.sbatch
#SBATCH --job-name=taxi-robust-boot
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:50:00
#SBATCH --array=0-39
#SBATCH --requeue

set -euo pipefail

# Sizes from the test job 66191143 (debug, 2 CPUs, all-hours sample, 2026-11-15): setup
# (data, point estimate, diagnostics) 2.7 min, 61 s per draw per CPU, peak 5.5 GB with two
# workers. Draws scale with n: about 27 s for the daytime rows (n = 633,616) and 23 s for the
# $5-$25 sample. So the all-hours row gets 8 shards of 125 draws and the others 4 of 250,
# about 15-20 minutes each on 8 CPUs; the limit allows 2.5 times that for slow nodes.
# About 90 CPU-hours in all (about $1.35 at 2504.5 billing units per CPU-minute).
#
# Row name | draws per shard | arguments that refit the row's design variation.
ROWS=(
  "allhours|125|--sample allhours"
  "eps0.075|250|--eps 0.075"
  "eps0.10|250|--eps 0.10"
  "Jtail0.0005|250|--tail 0.0005"
  "Jtail0.001|250|--tail 0.001"
  "Jtail0.004|250|--tail 0.004"
  "Jwin0.001-0.002|250|--tail 0.001 --window-tail 0.002"
  "Jwin0.005-0.01|250|--tail 0.005 --window-tail 0.01"
  "restricted|250|--sample restricted"
)
BOOT=999
TASKS=()                            # one entry per shard: row index and first draw
for ((r = 0; r < ${#ROWS[@]}; r++)); do
    IFS='|' read -r _ per _ <<< "${ROWS[$r]}"
    for ((a = 0; a < BOOT; a += per)); do TASKS+=("$r $a $per"); done
done
if (( SLURM_ARRAY_TASK_ID >= ${#TASKS[@]} )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} has no shard (only ${#TASKS[@]})" >&2
    exit 1
fi
read -r ROW A PER <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
IFS='|' read -r NAME _ ARGSTR <<< "${ROWS[$ROW]}"
read -r -a ARGS <<< "${ARGSTR}"
B=$(( A + PER < BOOT ? A + PER : BOOT ))

OUT="${OUT:?set OUT to the run directory}"
SHARD_OUT="${OUT}/${NAME}/shards/b${A}-${B}"
mkdir -p "${SHARD_OUT}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH="$PWD" MPLBACKEND=Agg
PY="${PY:-$HOME/envs/perfrdd/bin/python}"

{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
    echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"
    echo "row=${NAME} args=${ARGS[*]} draws=${A}..$((B - 1))"
    echo "commit=$(git rev-parse HEAD) python=$("$PY" --version 2>&1)"
    git status --short --untracked-files=no
} > "${SHARD_OUT}/provenance.txt"

srun --cpu-bind=cores "$PY" -m experiments.scripts.taxi_section6 --specs R2-E1 "${ARGS[@]}" \
    --no-robust --boot-start "$A" --boot "$B" --workers "${SLURM_CPUS_PER_TASK}" --out "${SHARD_OUT}"
