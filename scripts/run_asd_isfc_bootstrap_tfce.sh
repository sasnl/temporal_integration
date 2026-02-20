#!/bin/bash
# Run ISFC Bootstrap TFCE for ASD group — one job at a time
# 3 conditions x 3 seeds = 9 jobs

TI_CODE="/Users/tongshan/Documents/TemporalIntegration/code/TI_code"
DATA_DIR="/Users/tongshan/Documents/TemporalIntegration/data/asd_ti_processed"
RESULT_BASE="/Users/tongshan/Documents/TemporalIntegration/result/ASD/ISFC/bootstrap/tfce/loo"
N_PERMS=1000

cd "$TI_CODE"
mkdir -p "$RESULT_BASE/PMC" "$RESULT_BASE/LpSTS" "$RESULT_BASE/RpSTS"

CONDITIONS=("TI1_orig" "TI1_sent" "TI1_word")

# Seeds: name, x, y, z, output_subdir
SEEDS=(
    "PMC 0 -53 2 PMC"
    "LpSTS -63 -42 9 LpSTS"
    "RpSTS 57 -31 5 RpSTS"
)

TOTAL=0
PASSED=0

for seed_info in "${SEEDS[@]}"; do
    read -r seed_name sx sy sz out_sub <<< "$seed_info"
    for condition in "${CONDITIONS[@]}"; do
        TOTAL=$((TOTAL + 1))
        echo ""
        echo "============================================"
        echo "  ISFC Bootstrap TFCE: $seed_name / $condition"
        echo "  Output: $RESULT_BASE/$out_sub"
        echo "  Started: $(date)"
        echo "============================================"

        python isfc/run_isfc_pipeline.py \
            --condition "$condition" \
            --isfc_method loo \
            --stats_method bootstrap \
            --n_perms "$N_PERMS" \
            --p_threshold 0.05 \
            --use_tfce \
            --seed_x "$sx" --seed_y "$sy" --seed_z "$sz" --seed_radius 5 \
            --data_dir "$DATA_DIR" \
            --output_dir "$RESULT_BASE/$out_sub"

        if [ $? -eq 0 ]; then
            PASSED=$((PASSED + 1))
            echo "  Completed: $seed_name / $condition at $(date)"
        else
            echo "  FAILED: $seed_name / $condition at $(date)"
            echo "  Stopping. $PASSED/$TOTAL passed."
            exit 1
        fi
    done
done

echo ""
echo "=== All done: $PASSED/$TOTAL passed ==="
