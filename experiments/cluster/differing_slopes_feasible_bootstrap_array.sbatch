#!/bin/bash
#SBATCH --job-name=ds-bootstrap
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail

N_VALUES=(800 1600 3200)
N_VALUE="${N_VALUES[${SLURM_ARRAY_TASK_ID}]}"
OUTER_REPS="${PERFRDD_OUTER_REPS:-50}"
BOOTSTRAP_REPS="${PERFRDD_BOOTSTRAP_REPS:-199}"
cd "${PERFRDD_REPO:?set PERFRDD_REPO to the repository root}"
OUT_ROOT="${PERFRDD_OUT:?set PERFRDD_OUT to an output directory}"
CELL_OUT="${OUT_ROOT}/n_${N_VALUE}"
mkdir -p "${CELL_OUT}"

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
    echo "sample_size=${N_VALUE}"
    echo "outer_replications=${OUTER_REPS}"
    echo "bootstrap_replications=${BOOTSTRAP_REPS}"
    echo "commit=$(git rev-parse HEAD)"
    echo "python=$(python3 --version 2>&1)"
    git status --short --branch
} > "${CELL_OUT}/provenance.txt"

srun --cpu-bind=cores python3 -m experiments.scripts.differing_slopes_feasible_bootstrap \
  --scenario baseline \
  --n "${N_VALUE}" \
  --outer-reps "${OUTER_REPS}" \
  --bootstrap-reps "${BOOTSTRAP_REPS}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --variant oracle full_spline_ols full_spline_ridge \
  --seed 20260926 \
  --out "${CELL_OUT}"
