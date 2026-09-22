#!/bin/bash
#SBATCH --job-name=perfrdd-bootstrap
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --array=0-3%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue

set -euo pipefail

N_VALUES=(1200 2400 4800 9600)
N_VALUE="${N_VALUES[${SLURM_ARRAY_TASK_ID}]}"
cd "${PERFRDD_REPO:?set PERFRDD_REPO to the repository root}"
OUT_ROOT="${PERFRDD_OUT:?set PERFRDD_OUT to an output directory}"
CELL_OUT="${OUT_ROOT}/n_${N_VALUE}"
mkdir -p "${CELL_OUT}"

# One numerical-library thread per Python worker avoids silently oversubscribing
# the Slurm allocation (the Python process pool supplies the parallelism).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONPATH="${PERFRDD_REPO}"

# Keep enough provenance to reproduce or diagnose an individual array cell.
{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname=$(hostname)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "array_job_id=${SLURM_ARRAY_JOB_ID}"
    echo "array_task_id=${SLURM_ARRAY_TASK_ID}"
    echo "sample_size=${N_VALUE}"
    echo "commit=$(git rev-parse HEAD)"
    echo "python=$(python3 --version 2>&1)"
    git status --short --branch
} > "${CELL_OUT}/provenance.txt"

srun --cpu-bind=cores python3 -m experiments.scripts.hard_trim_bootstrap_active \
  --n "${N_VALUE}" \
  --outer-reps 50 \
  --bootstrap-reps 199 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --density gaussian \
  --seed 20260922 \
  --resume \
  --out "${CELL_OUT}"
