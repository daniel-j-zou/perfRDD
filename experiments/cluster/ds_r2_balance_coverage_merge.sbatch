#!/bin/bash
# Merge the shards of ds_r2_balance_coverage_array.sbatch (both sets) into one run directory
# and write <design>_summary.json. Submit with
#   --dependency=afterok:<long id>:<short id> --export=ALL,OUT=<run directory>
# It refuses to summarize an incomplete cell, because the coverage script would otherwise
# start computing the missing replications inside this small job.
#SBATCH --job-name=ds-r2bal-merge
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
OUT="${OUT:?set OUT to the run directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"
REPS=500
DESIGNS="${DESIGNS:-r2_10_bal r2_30_bal r2_50_bal r2_70_bal r2_90_bal r2_10_unbal r2_30_unbal r2_50_unbal r2_70_unbal r2_90_unbal}"
NS="${NS:-4000 16000 64000 256000}"

bad=0
for d in $DESIGNS; do
    for n in $NS; do
        target="${OUT}/${d}_n${n}.jsonl"
        cat "${OUT}"/shards/${d}_n${n}_r*/${d}_n${n}.jsonl > "${target}"
        got=$(cut -c1-40 "${target}" | grep -c '"rep"' || true)
        uniq=$(cut -c1-40 "${target}" | sort -u | grep -c '"rep"' || true)
        echo "${d} n=${n}: ${got} rows, ${uniq} distinct, of ${REPS}"
        if (( got != REPS || uniq != REPS )); then
            echo "incomplete or duplicated cell ${d} n=${n}" >&2
            bad=1
        fi
    done
done
(( bad == 0 )) || { echo "not summarizing" >&2; exit 1; }

for d in $DESIGNS; do
    "$PY" -m experiments.scripts.ds_r2_balance_coverage --design "$d" --n $NS --reps $REPS \
        --boot 999 --workers 1 --out "$OUT"
done
