#!/bin/bash
# Merge the shards of taxi_robust_boot_array.sbatch: for each row, concatenate the shards'
# R2-E1_draws.jsonl, check that all 999 draws are there, and rerun taxi_section6 with the full
# --boot (every draw done, so it only refits the point estimate and writes summary.json).
# Submit with --dependency=afterok:<array job id> --export=ALL,OUT=<run directory>.
#SBATCH --job-name=taxi-robust-merge
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00

set -euo pipefail
OUT="${OUT:?set OUT to the run directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" MPLBACKEND=Agg
PY="${PY:-$HOME/envs/perfrdd/bin/python}"
BOOT=999
declare -A ARGS=(
  [allhours]="--sample allhours" [eps0.075]="--eps 0.075" [eps0.10]="--eps 0.10"
  [Jtail0.0005]="--tail 0.0005" [Jtail0.001]="--tail 0.001" [Jtail0.004]="--tail 0.004"
  [Jwin0.001-0.002]="--tail 0.001 --window-tail 0.002"
  [Jwin0.005-0.01]="--tail 0.005 --window-tail 0.01" [restricted]="--sample restricted"
)
status=0
for NAME in "${!ARGS[@]}"; do
    dir="${OUT}/${NAME}"
    cat "${dir}"/shards/b*/R2-E1_draws.jsonl > "${dir}/R2-E1_draws.jsonl"
    got=$(grep -c '"b"' "${dir}/R2-E1_draws.jsonl" || true)
    echo "${NAME}: ${got} of ${BOOT} draws"
    if (( got != BOOT )); then
        echo "incomplete row ${NAME}; not summarizing" >&2
        status=1
        continue
    fi
    read -r -a A <<< "${ARGS[$NAME]}"
    "$PY" -m experiments.scripts.taxi_section6 --specs R2-E1 "${A[@]}" --no-robust \
        --boot "$BOOT" --workers 1 --out "${dir}"
done
exit $status
