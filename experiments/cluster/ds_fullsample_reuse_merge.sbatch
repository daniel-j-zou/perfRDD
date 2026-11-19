#!/bin/bash
# Merge the shards of ds_fullsample_reuse_array.sbatch and write full_summary.json; refuses an
# incomplete cell. Submit with --dependency=afterok:<array id> --export=ALL,OUT=<run dir>.
#SBATCH --job-name=ds-fullsample-merge
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail
OUT="${OUT:?set OUT to the run directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"
bad=0
for n in 4000 16000 64000; do
    target="${OUT}/full_n${n}.jsonl"
    cat "${OUT}"/shards/full_n${n}_r*/full_n${n}.jsonl > "${target}"
    got=$(cut -c1-30 "${target}" | grep -c '"rep"' || true)
    uniq=$(cut -c1-30 "${target}" | sort -u | grep -c '"rep"' || true)
    echo "n=${n}: ${got} rows, ${uniq} distinct, of 500"
    (( got == 500 && uniq == 500 )) || { echo "incomplete cell n=${n}" >&2; bad=1; }
done
(( bad == 0 )) || { echo "not summarizing" >&2; exit 1; }
"$PY" -m experiments.scripts.ds_fullsample_reuse --n 4000 16000 64000 --reps 500 --boot 999 \
    --workers 1 --out "$OUT"
