#!/bin/bash
# Run ISFC Contrast (LOO, Bootstrap TFCE) for ASD group
# 3 contrasts x 3 seeds = 9 jobs, sequential

TI_CODE="/Users/tongshan/Documents/TemporalIntegration/code/TI_code"
DATA_DIR="/Users/tongshan/Documents/TemporalIntegration/data/asd_ti_processed"
N_PERMS=1000

cd "$TI_CODE"

CONTRASTS=("TI1_orig TI1_sent" "TI1_orig TI1_word" "TI1_sent TI1_word")
SEEDS=(
    "PMC seed0_-53_2_r5 PMC"
    "LpSTS seed-63_-42_9_r5 LpSTS"
    "RpSTS seed57_-31_5_r5 RpSTS"
)

TOTAL=0
PASSED=0

for seed_info in "${SEEDS[@]}"; do
    read -r seed_label seed_name out_sub <<< "$seed_info"
    ISFC_DIR="/Users/tongshan/Documents/TemporalIntegration/result/ASD/ISFC/bootstrap/tfce/loo/$out_sub"
    OUT_DIR="/Users/tongshan/Documents/TemporalIntegration/result/ASD/ISFC_contrast/bootstrap_tfce/loo/$out_sub"
    mkdir -p "$OUT_DIR"

    for contrast in "${CONTRASTS[@]}"; do
        read -r cond1 cond2 <<< "$contrast"
        TOTAL=$((TOTAL + 1))
        echo ""
        echo "============================================"
        echo "  ISFC Contrast: $cond1 vs $cond2 / $seed_label"
        echo "  Started: $(date)"
        echo "============================================"

        python shared/run_contrast.py \
            --cond1 "$cond1" --cond2 "$cond2" \
            --type isfc \
            --isc_method loo \
            --seed_name "$seed_name" \
            --method bootstrap \
            --n_perms "$N_PERMS" \
            --use_tfce \
            --data_dir "$ISFC_DIR" \
            --output_dir "$OUT_DIR" \
            --auto_subjects_dir "$DATA_DIR"

        if [ $? -eq 0 ]; then
            PASSED=$((PASSED + 1))
            echo "  Completed: $seed_label / $cond1 vs $cond2 at $(date)"
        else
            echo "  FAILED: $seed_label / $cond1 vs $cond2 at $(date)"
            echo "  Stopping. $PASSED/$TOTAL passed."
            exit 1
        fi
    done
done

echo ""
echo "=== All done: $PASSED/$TOTAL passed ==="
