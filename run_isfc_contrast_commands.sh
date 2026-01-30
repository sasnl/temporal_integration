#!/bin/bash
# Run ISFC Contrast Analysis with Bootstrap & TFCE
# Seeds: PMC, Left pSTS, Right pSTS

DATA_BASE="/Users/tongshan/Documents/TemporalIntegration/result/ISFC/bootstrap/tfce"

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
            --method bootstrap --n_perms 1000 --use_tfce \


        # 2. Orig vs Word
        python shared/run_contrast.py \
            --data_dir "$DATA_DIR" \
            --cond1 TI1_orig --cond2 TI1_word \
            --type isfc \
            --isc_method "$METHOD" \
            --seed_name "$SEED" \
            --method bootstrap --n_perms 1000 --use_tfce \

        # 3. Sent vs Word
        python shared/run_contrast.py \
            --data_dir "$DATA_DIR" \
            --cond1 TI1_sent --cond2 TI1_word \
            --type isfc \
            --isc_method "$METHOD" \
            --seed_name "$SEED" \
            --method bootstrap --n_perms 1000 --use_tfce \
            
    done
done

echo "All ISFC contrast analyses complete."
