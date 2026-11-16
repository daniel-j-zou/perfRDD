#!/bin/bash
# Merge the shards of ds_section5_coverage_array.sbatch into one run directory and write the
# summary (displayed_summary.json), as one unsplit run would. Submit with
#   --dependency=afterok:<array job id> --export=ALL,OUT=<run directory>
# It refuses to summarize an incomplete cell, because the coverage script would otherwise
# start computing the missing replications inside this small job. R64 sets the number of
# data sets expected at n = 64,000 (default 200; 500 after ds_section5_64k_extend_array.sbatch).
#SBATCH --job-name=ds5-merge
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00

set -euo pipefail
OUT="${OUT:?set OUT to the run directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"

R64="${R64:-200}"
declare -A REPS=([4000]=500 [16000]=500 [64000]=${R64})
for N in 4000 16000 64000; do
    target="${OUT}/displayed_n${N}.jsonl"
    cat "${OUT}"/shards/n${N}_r*/displayed_n${N}.jsonl > "${target}"
    got=$(cut -c1-40 "${target}" | grep -c '"rep"' || true)
    echo "n=${N}: ${got} of ${REPS[$N]} replications"
    if (( got != REPS[$N] )); then
        echo "incomplete cell n=${N}; not summarizing" >&2
        exit 1
    fi
done

"$PY" -m experiments.scripts.ds_four_role_coverage --design displayed --n 4000 16000 \
    --reps 500 --boot 999 --workers 1 --out "${OUT}"
"$PY" -m experiments.scripts.ds_four_role_coverage --design displayed --n 64000 \
    --reps "${R64}" --boot 999 --workers 1 --out "${OUT}"
"$PY" -m experiments.scripts.ds_four_role_coverage --design displayed --report --out "${OUT}"
