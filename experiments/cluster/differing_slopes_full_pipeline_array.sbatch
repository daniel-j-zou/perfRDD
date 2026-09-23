#!/bin/bash
#SBATCH --job-name=ds-feasible
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

SCENARIOS=(baseline quadratic_misspecification weak_curvature boundary_upper clustered)
SCENARIO="${SCENARIOS[${SLURM_ARRAY_TASK_ID}]}"
REPS="${PERFRDD_REPS:-100}"
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

srun --cpu-bind=cores python3 -m experiments.scripts.differing_slopes_full_pipeline \
  --scenario "${SCENARIO}" \
  --n 800 1600 3200 6400 \
  --reps "${REPS}" \
  --variant oracle full_spline_ols full_spline_ridge \
  --seed 20260926 \
  --out "${OUT_ROOT}/${SCENARIO}.json"
