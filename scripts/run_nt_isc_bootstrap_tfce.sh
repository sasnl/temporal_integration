#!/bin/bash
# Script to rerun ISC Bootstrap with TFCE for NT group
# Runs ONE method + ONE condition at a time to avoid memory issues
#
# Usage:
#   bash scripts/run_nt_isc_bootstrap_tfce.sh <method> <condition>
#
# Examples:
#   bash scripts/run_nt_isc_bootstrap_tfce.sh loo TI1_orig
#   bash scripts/run_nt_isc_bootstrap_tfce.sh pairwise TI1_sent
#   bash scripts/run_nt_isc_bootstrap_tfce.sh all all       # run all 6 sequentially
#
# Methods:  loo, pairwise, all
# Conditions: TI1_orig, TI1_sent, TI1_word, all

TI_CODE="/Users/tongshan/Documents/TemporalIntegration/code/TI_code"
DATA_DIR="/Users/tongshan/Documents/TemporalIntegration/data/nt_ti_processed"
RESULT_BASE="/Users/tongshan/Documents/TemporalIntegration/result/NT/ISC_bootstrap/tfce"
N_PERMS=1000

METHOD="${1:-all}"
CONDITION="${2:-all}"

cd "$TI_CODE"

run_one() {
    local method="$1"
    local condition="$2"
    local out_dir="$RESULT_BASE/$method"

    mkdir -p "$out_dir"

    echo ""
    echo "============================================"
    echo "  ISC Bootstrap TFCE: $method / $condition"
    echo "  Output: $out_dir"
    echo "  Started: $(date)"
    echo "============================================"

    python isc/run_isc_pipeline.py \
        --condition "$condition" \
        --isc_method "$method" \
        --stats_method bootstrap \
        --n_perms "$N_PERMS" \
        --p_threshold 0.05 \
        --use_tfce \
        --data_dir "$DATA_DIR" \
        --output_dir "$out_dir"

    local status=$?
    if [ $status -eq 0 ]; then
        echo "  Completed successfully: $method / $condition at $(date)"
    else
        echo "  FAILED: $method / $condition (exit code $status) at $(date)"
        return $status
    fi
}

# Build list of methods
if [ "$METHOD" = "all" ]; then
    METHODS=("loo" "pairwise")
else
    METHODS=("$METHOD")
fi

# Build list of conditions
if [ "$CONDITION" = "all" ]; then
    CONDITIONS=("TI1_orig" "TI1_sent" "TI1_word")
else
    CONDITIONS=("$CONDITION")
fi

echo "=== NT ISC Bootstrap TFCE ==="
echo "Methods: ${METHODS[*]}"
echo "Conditions: ${CONDITIONS[*]}"
echo "N permutations: $N_PERMS"
echo "Data dir: $DATA_DIR"
echo "Output base: $RESULT_BASE"
echo ""

TOTAL=0
PASSED=0
FAILED=0

for method in "${METHODS[@]}"; do
    for condition in "${CONDITIONS[@]}"; do
        TOTAL=$((TOTAL + 1))
        run_one "$method" "$condition"
        if [ $? -eq 0 ]; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
            echo "Stopping due to failure. Fix the issue and rerun remaining jobs."
            echo "Summary: $PASSED/$TOTAL passed, $FAILED failed"
            exit 1
        fi
    done
done

echo ""
echo "=== All done: $PASSED/$TOTAL passed ==="

# Rethreshold
echo "Running rethresholding..."
python shared/rethreshold_results.py --result_dir "$RESULT_BASE" --thresholds 0.01 0.005 0.001 --overwrite
echo "Done."
