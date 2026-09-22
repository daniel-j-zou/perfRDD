#!/bin/bash
#SBATCH --job-name=perfrdd-bootstrap
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=active-bootstrap-%j.out
#SBATCH --error=active-bootstrap-%j.err

set -euo pipefail

cd "${PERFRDD_REPO:?set PERFRDD_REPO to the repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONPATH="${PERFRDD_REPO}"

srun --cpu-bind=cores python3 -m experiments.scripts.hard_trim_bootstrap_active \
  --n 1200 2400 4800 9600 \
  --outer-reps 50 \
  --bootstrap-reps 199 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --density gaussian \
  --seed 20260922 \
  --resume \
  --out "${PERFRDD_OUT:?set PERFRDD_OUT to an output directory}"
