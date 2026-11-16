#!/bin/bash
# Appendix 7 "Non-Gaussian linear predictor" (cited in §5.2): the four-role estimator in the t5
# and mixture designs, rerun with the nonlinear effect of the Section 5 design
# (alpha = 2 + 0.4 eta + 0.8 tanh(1.5 eta), b = eta^2/2 + 0.3 sin(2 eta)); the 2026-11-09 run
# used alpha = 2 + eta. Same plan as that run: 800 / 400 / 200 data sets at n = 4k / 16k / 64k
# for both laws, then 200 mixture data sets at n = 256k. Point estimates only (no bootstrap).
# About 1.4 CPU-hours (from the 2026-11-09 local run, scaled by the cluster's speed), so one
# 8-CPU task of about 10 minutes; peak memory is a few hundred MB per worker at n = 256k.
#
# Submit from the repository root on Great Lakes:
#   OUT=experiments/runs/four_role_nongaussian_nl; mkdir -p $OUT
#   sbatch --output=$OUT/%x_%j.out --export=ALL,OUT=$OUT experiments/cluster/ds_nongaussian_nl.sbatch
#SBATCH --job-name=ds-nongauss-nl
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:30:00

set -euo pipefail
OUT="${OUT:?set OUT to the run directory}"
export DS_NG_EFFECT=nonlinear
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0 PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"
W="${SLURM_CPUS_PER_TASK}"

{
    echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname) job=${SLURM_JOB_ID} cpus=${W}"
    echo "commit=$(git rev-parse HEAD) python=$("$PY" --version 2>&1) effect=${DS_NG_EFFECT}"
    git status --short --untracked-files=no
} > "${OUT}/provenance.txt"

M=experiments.scripts.ds_four_role_nongaussian
srun --cpu-bind=cores "$PY" -m $M --n 4000 16000 64000 --reps 800 400 200 --workers "$W" --out "$OUT"
srun --cpu-bind=cores "$PY" -m $M --laws mixture --n 256000 --reps 200 --workers "$W" --out "$OUT"
"$PY" -m $M --summarize --out "$OUT"
