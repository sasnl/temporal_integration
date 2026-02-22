#!/bin/bash
# Run NT vs ASD group comparison — ISC + ISFC, loo, sequential
# ISC: 3 conditions = 3 jobs
# ISFC: 3 conditions x 3 seeds = 9 jobs
# Total: 12 jobs

TI_CODE="/Users/tongshan/Documents/TemporalIntegration/code/TI_code"
NT_ISC_DIR="/Users/tongshan/Documents/TemporalIntegration/result/NT/ISC/bootstrap/tfce/loo"
ASD_ISC_DIR="/Users/tongshan/Documents/TemporalIntegration/result/ASD/ISC/bootstrap/tfce/loo"
OUTPUT_DIR="/Users/tongshan/Documents/TemporalIntegration/result/group_comparison"
N_PERMS=1000

cd "$TI_CODE"

CONDITIONS=("TI1_orig" "TI1_sent" "TI1_word")
TOTAL=0
PASSED=0

# ==========================================
# ISC Group Comparison
# ==========================================
echo "=== ISC Group Comparisons ==="

for condition in "${CONDITIONS[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo ""
    echo "============================================"
    echo "  Group ISC: $condition"
    echo "  Started: $(date)"
    echo "============================================"

    python shared/run_group_comparison.py \
        --condition "$condition" \
        --analysis_type isc \
        --method loo \
        --nt_result_dir "$NT_ISC_DIR" \
        --asd_result_dir "$ASD_ISC_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --n_perms "$N_PERMS"

    if [ $? -eq 0 ]; then
        PASSED=$((PASSED + 1))
    else
        echo "  FAILED: ISC $condition"
        exit 1
    fi
done

# ==========================================
# ISFC Group Comparison
# ==========================================
echo ""
echo "=== ISFC Group Comparisons ==="

SEEDS=(
    "PMC seed0_-53_2_r5 PMC"
    "LpSTS seed-63_-42_9_r5 LpSTS"
    "RpSTS seed57_-31_5_r5 RpSTS"
)

for seed_info in "${SEEDS[@]}"; do
    read -r seed_label seed_name out_sub <<< "$seed_info"
    NT_ISFC_DIR="/Users/tongshan/Documents/TemporalIntegration/result/NT/ISFC/bootstrap/tfce/loo/$out_sub"
    ASD_ISFC_DIR="/Users/tongshan/Documents/TemporalIntegration/result/ASD/ISFC/bootstrap/tfce/loo/$out_sub"

    for condition in "${CONDITIONS[@]}"; do
        TOTAL=$((TOTAL + 1))
        echo ""
        echo "============================================"
        echo "  Group ISFC: $condition / $seed_label"
        echo "  Started: $(date)"
        echo "============================================"

        python shared/run_group_comparison.py \
            --condition "$condition" \
            --analysis_type isfc \
            --method loo \
            --seed_name "$seed_name" \
            --nt_result_dir "$NT_ISFC_DIR" \
            --asd_result_dir "$ASD_ISFC_DIR" \
            --output_dir "$OUTPUT_DIR" \
            --n_perms "$N_PERMS"

        if [ $? -eq 0 ]; then
            PASSED=$((PASSED + 1))
        else
            echo "  FAILED: ISFC $condition / $seed_label"
            exit 1
        fi
    done
done

echo ""
echo "=== All done: $PASSED/$TOTAL passed ==="
