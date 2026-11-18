#!/bin/bash
# Merge the shards of ds_review_ablations_array.sbatch (both sets) into one run directory
# and write <design>_summary.json. Submit with
#   --dependency=afterok:<long id>:<short id> --export=ALL,OUT=<run directory>
# It refuses to summarize an incomplete cell, because the script would otherwise start
# computing the missing replications inside this small job. DESIGNS and LONG_DESIGNS
# can be overridden through --export to merge a subset.
#SBATCH --job-name=ds-review-merge
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:30:00

set -euo pipefail
OUT="${OUT:?set OUT to the run directory}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD"
PY="${PY:-$HOME/envs/perfrdd/bin/python}"
REPS=500
DESIGNS="${DESIGNS:-cluster_g31 interact_k10 interact_k20 grid_h27 flat_l50 flat_l25 skew hetero_a148 hetero_b148 hetero_ca hetero_cb}"
LONG_DESIGNS="${LONG_DESIGNS:-cluster_g31 interact_k10 interact_k20 hetero_a148 hetero_b148 hetero_ca hetero_cb}"

ns_of() {   # sample sizes of a design: 256k only for the long designs (499 draws there)
    case " ${LONG_DESIGNS} " in *" $1 "*) echo "16000 64000 256000" ;; *) echo "16000 64000" ;; esac
}

bad=0
for d in $DESIGNS; do
    for n in $(ns_of "$d"); do
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

# --boot only matters for missing replications (none here); each cell records the number
# of draws per data set it actually has (999 at 16k and 64k, 499 at 256k).
for d in $DESIGNS; do
    "$PY" -m experiments.scripts.ds_review_ablations --design "$d" --n $(ns_of "$d") \
        --reps $REPS --boot 999 --workers 1 --out "$OUT"
done
