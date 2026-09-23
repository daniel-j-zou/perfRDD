#!/bin/bash
#SBATCH --job-name=ds-distribution
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --array=0-8%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail

SCENARIOS=(
  baseline x_t5 x_skewed x_mixture
  eta_t5 eta_skewed
  error_t5 error_skewed error_heteroskedastic
)
SCENARIO="${SCENARIOS[${SLURM_ARRAY_TASK_ID}]}"
REPS="${PERFRDD_REPS:-250}"
cd "${PERFRDD_REPO:?set PERFRDD_REPO to the repository root}"
OUT_ROOT="${PERFRDD_OUT:?set PERFRDD_OUT to an output directory}"
mkdir -p "${OUT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONPATH="${PERFRDD_REPO}"

{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "scenario=${SCENARIO}"
    echo "replications=${REPS}"
    echo "commit=$(git rev-parse HEAD)"
    echo "python=$(python3 --version 2>&1)"
    git status --short --branch
} > "${OUT_ROOT}/provenance_${SCENARIO}.txt"

srun --cpu-bind=cores python3 -m experiments.scripts.differing_slopes_distributional_battery \
  --scenario "${SCENARIO}" \
  --n 1200 2400 4800 \
  --reps "${REPS}" \
  --seed 20260923 \
  --out "${OUT_ROOT}/${SCENARIO}.json"
