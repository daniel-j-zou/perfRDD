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
#SBATCH --mem-per-cpu=2G
#SBATCH --time=01:00:00
#SBATCH --array=0-35
#SBATCH --requeue

set -euo pipefail

# Row name and the arguments that refit its design variation.
ROWS=(
  "allhours|--sample allhours"
  "eps0.075|--eps 0.075"
  "eps0.10|--eps 0.10"
  "Jtail0.0005|--tail 0.0005"
  "Jtail0.001|--tail 0.001"
  "Jtail0.004|--tail 0.004"
  "Jwin0.001-0.002|--tail 0.001 --window-tail 0.002"
  "Jwin0.005-0.01|--tail 0.005 --window-tail 0.01"
  "restricted|--sample restricted"
)
BOOT=999
PER_SHARD=250                       # 4 shards per row: 0-249, 250-499, 500-749, 750-998
SHARDS_PER_ROW=$(( (BOOT + PER_SHARD - 1) / PER_SHARD ))
ROW=$(( SLURM_ARRAY_TASK_ID / SHARDS_PER_ROW ))
SHARD=$(( SLURM_ARRAY_TASK_ID % SHARDS_PER_ROW ))
if (( ROW >= ${#ROWS[@]} )); then
    echo "array index ${SLURM_ARRAY_TASK_ID} has no row" >&2
    exit 1
fi
NAME="${ROWS[$ROW]%%|*}"
read -r -a ARGS <<< "${ROWS[$ROW]#*|}"
A=$(( SHARD * PER_SHARD ))
B=$(( A + PER_SHARD < BOOT ? A + PER_SHARD : BOOT ))

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
