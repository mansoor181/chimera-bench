#!/bin/bash
# ===========================================================================
# Train and evaluate dyMEAN across all splits using CHIMERA.
#
# dyMEAN is a multi-CDR baseline that generates all CDRs simultaneously.
# After each split finishes, its evaluation results are appended to a
# unified CSV (same format as run_all_baselines.sh).
#
# Usage:
#   GPU=0 bash chimera_train.sh                         # all splits
#   GPU=0 SPLITS="epitope_group" bash chimera_train.sh   # single split
#   GPU=0 MAX_EPOCH=50 bash chimera_train.sh             # override epochs
#   nohup GPU=0 bash chimera_train.sh > dymean_run.log 2>&1 &
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASELINE="dymean"
SPLITS=(${SPLITS:-epitope_group antigen_fold temporal})
GPU=${GPU:-0}
MAX_EPOCH=${MAX_EPOCH:-250}

RESULTS_CSV="${SCRIPT_DIR}/${BASELINE}_results.csv"
LOGDIR="${SCRIPT_DIR}/_logs"
mkdir -p "$LOGDIR"

# Resolve results_root once
RESULTS_ROOT=$(python -c "
import yaml, os
with open(os.path.join('${SCRIPT_DIR}', '..', 'shared_config.yaml')) as f:
    print(yaml.safe_load(f)['paths']['results_root'])
")

# Write CSV header
cat > "$RESULTS_CSV" <<'HEADER'
baseline,split,cdr_type,aar,caar,ppl,rmsd,tm_score,fnat,irmsd,dockq,epitope_f1,n_liabilities,chimera_s,chimera_b
HEADER

echo "=== dyMEAN CHIMERA-Bench Training ==="
echo "GPU=${GPU}, SPLITS=${SPLITS[*]}, MAX_EPOCH=${MAX_EPOCH}"
echo "Results CSV: ${RESULTS_CSV}"
echo "Logs: ${LOGDIR}/"
echo ""

eval_and_append() {
    local split="$1"
    local label="${BASELINE}_all_${split}"
    local pred_dir="${RESULTS_ROOT}/dymean/${label}/predictions"
    local eval_logfile="${LOGDIR}/${label}_eval.log"

    if [[ ! -d "$pred_dir" ]]; then
        echo "[$(date +%H:%M:%S)] WARN   No predictions dir for ${label}: ${pred_dir}"
        return 0
    fi

    # Use --predictions for multi-CDR baselines (finds H1/H2/.../L3 subdirs)
    python chimera_evaluate.py --predictions "$pred_dir" --split "$split" \
        > "$eval_logfile" 2>&1 || true

    local eval_csv
    eval_csv="$(dirname "$pred_dir")/eval_metrics.csv"

    if [[ ! -f "$eval_csv" ]]; then
        echo "[$(date +%H:%M:%S)] WARN   No eval CSV found for ${label}"
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
    label="${BASELINE}_all_${split}"
    logfile="${LOGDIR}/${label}.log"

    echo "[$(date +%H:%M:%S)] (${DONE}/${TOTAL}) Training: split=${split} (all CDRs)"
    if python chimera_trainer.py \
            --split "$split" \
            --gpu "$GPU" --max_epoch "$MAX_EPOCH" \
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
echo "dyMEAN training complete."
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
