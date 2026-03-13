#!/bin/bash
# ===========================================================================
# Train and evaluate RADAb across all splits using CHIMERA.
#
# RADAb generates all 6 CDRs simultaneously (retrieval-augmented diffusion),
# so we train once per split (not per CDR). Evaluation is done per-CDR from
# the combined output.
#
# After each experiment finishes, its evaluation results are appended to a
# unified CSV (same format as run_all_baselines.sh).
#
# Usage:
#   GPU=0 bash chimera_train.sh                         # all splits
#   GPU=0 SPLITS="epitope_group" bash chimera_train.sh   # single split
#   nohup GPU=0 bash chimera_train.sh > radab_run.log 2>&1 &
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASELINE="radab"
SPLITS=(${SPLITS:-epitope_group antigen_fold temporal})
GPU=${GPU:-0}

RESULTS_CSV="${SCRIPT_DIR}/${BASELINE}_results.csv"
LOGDIR="${SCRIPT_DIR}/_logs"
mkdir -p "$LOGDIR"

# Write CSV header
cat > "$RESULTS_CSV" <<'HEADER'
baseline,split,cdr_type,aar,caar,ppl,rmsd,tm_score,fnat,irmsd,dockq,epitope_f1,n_liabilities,chimera_s,chimera_b
HEADER

echo "=== RADAb CHIMERA-Bench Training ==="
echo "GPU=${GPU}, SPLITS=${SPLITS[*]}"
echo "Results CSV: ${RESULTS_CSV}"
echo "Logs: ${LOGDIR}/"
echo ""

eval_and_append() {
    local split="$1"
    local eval_logfile="${LOGDIR}/${BASELINE}_${split}_eval.log"
    local eval_csv="${LOGDIR}/${BASELINE}_${split}_eval_metrics.csv"

    echo "[$(date +%H:%M:%S)] Evaluating ${BASELINE}, split=${split}..."
    python chimera_evaluate.py --aggregate --split "$split" --output "$eval_csv" \
        > "$eval_logfile" 2>&1 || true

    if [[ ! -f "$eval_csv" ]]; then
        echo "[$(date +%H:%M:%S)] WARN   No eval CSV found for ${BASELINE}/${split}"
        return 0
    fi

    # Append rows (skip header), prepend baseline+split.
    tail -n +2 "$eval_csv" | awk -v bl="$BASELINE" -v sp="$split" '{print bl "," sp "," $0}' >> "$RESULTS_CSV"
}

TOTAL=${#SPLITS[@]}
DONE=0
FAILED=0

for split in "${SPLITS[@]}"; do
    DONE=$((DONE + 1))
    label="${BASELINE}_multicdrs_${split}"
    logfile="${LOGDIR}/${label}.log"

    echo "[$(date +%H:%M:%S)] (${DONE}/${TOTAL}) Training: split=${split}"
    if python chimera_trainer.py \
            --split "$split" --gpu "$GPU" \
            > "$logfile" 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE   ${label}"
    else
        echo "[$(date +%H:%M:%S)] FAIL   ${label} (see ${logfile})"
        FAILED=$((FAILED + 1))
    fi

    eval_and_append "$split"
done

echo ""
echo "============================================"
echo "RADAb training complete."
echo "  Total: ${TOTAL}"
echo "  Failed: ${FAILED}"
echo "  Results: ${RESULTS_CSV}"
echo "============================================"

# Print summary
if [[ -f "$RESULTS_CSV" ]]; then
    echo ""
    echo "Results preview:"
    column -t -s, "$RESULTS_CSV" | head -20
fi
