#!/bin/bash
# Merge the shards of ds_nongaussian_coverage_array.sbatch (both sets) into one run directory
# and write <law>_summary.json. Submit with
#   --dependency=afterok:<long id>:<short id> --export=ALL,OUT=<run directory>
# It refuses to summarize an incomplete cell, because the coverage script would otherwise
# start computing the missing replications inside this small job.
#SBATCH --job-name=ds-ng-cov-merge
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00

set -euo pipefail
OUT="${OUT:?set OUT to the run directory}"
export DS_NG_EFFECT=nonlinear OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"
REPS=500

for cell in t5:4000 t5:16000 t5:64000 mixture:4000 mixture:16000 mixture:64000 mixture:256000; do
    law=${cell%%:*}; n=${cell##*:}
    target="${OUT}/${law}_n${n}.jsonl"
    cat "${OUT}"/shards/${law}_n${n}_r*/${law}_n${n}.jsonl > "${target}"
    got=$(cut -c1-40 "${target}" | grep -c '"rep"' || true)
    uniq=$(cut -c1-40 "${target}" | sort -u | grep -c '"rep"' || true)
    echo "${law} n=${n}: ${got} rows, ${uniq} distinct, of ${REPS}"
    if (( got != REPS || uniq != REPS )); then
        echo "incomplete or duplicated cell ${law} n=${n}; not summarizing" >&2
        exit 1
    fi
done

M=experiments.scripts.ds_nongaussian_coverage
"$PY" -m $M --law t5 --n 4000 16000 64000 --reps $REPS --boot 999 --workers 1 --out "$OUT"
"$PY" -m $M --law mixture --n 4000 16000 64000 256000 --reps $REPS --boot 999 --workers 1 \
    --out "$OUT"
