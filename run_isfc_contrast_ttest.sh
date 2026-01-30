#!/bin/bash
# Run ISFC Contrast Analysis with Uncorrected T-test
# Seeds: PMC, Left pSTS, Right pSTS

DATA_BASE="/Users/tongshan/Documents/TemporalIntegration/result/ISFC/bootstrap/tfce"
# Note: Data source is the same (Z-score maps), regardless of stat method used later.

SEEDS=("seed0_-53_2_r5" "seed-63_-42_9_r5" "seed57_-31_5_r5")
ISFC_METHODS=("loo" "pairwise")

for METHOD in "${ISFC_METHODS[@]}"; do
    DATA_DIR="${DATA_BASE}/${METHOD}"
    echo "Processing Method: $METHOD (Data: $DATA_DIR)"
    
    for SEED in "${SEEDS[@]}"; do
        echo "  Running Seed: $SEED"
        
        # 1. Orig vs Sent
        python shared/run_contrast.py \
            --data_dir "$DATA_DIR" \
            --cond1 TI1_orig --cond2 TI1_sent \
            --type isfc \
            --isc_method "$METHOD" \
            --seed_name "$SEED" \
            --method ttest \
            
        # 2. Orig vs Word
        python shared/run_contrast.py \
            --data_dir "$DATA_DIR" \
            --cond1 TI1_orig --cond2 TI1_word \
            --type isfc \
            --isc_method "$METHOD" \
            --seed_name "$SEED" \
            --method ttest \
            
        # 3. Sent vs Word
        python shared/run_contrast.py \
            --data_dir "$DATA_DIR" \
            --cond1 TI1_sent --cond2 TI1_word \
            --type isfc \
            --isc_method "$METHOD" \
            --seed_name "$SEED" \
            --method ttest \
            
    done
done

echo "All ISFC T-test contrast analyses complete."
