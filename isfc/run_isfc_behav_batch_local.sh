#!/bin/bash

# Batch script to run ISFC-Behavior Correlation for all 3 seeds and conditions
# Seeds:
# 1. Left HG: seed-63_-42_9_r5
# 2. PMC: seed0_-53_2_r5
# 3. Right HG: seed57_-31_5_r5

SEEDS=("seed-63_-42_9_r5" "seed0_-53_2_r5" "seed57_-31_5_r5")
CONDS=("TI1_orig" "TI1_sent" "TI1_word")
CSV="data/demographic/60483ASDSpeakerListe-TISubjectDemographic_DATA_2026-01-15_1602_combined.csv"
COL="srs_com_standard"
PERMS=1000
INPUT_DIR="result/ISFC/bootstrap/tfce/loo"

echo "Starting Batch ISFC Correlation Analysis (n=${PERMS})..."

for SEED in "${SEEDS[@]}"; do
    echo "==================================================="
    echo "Processing Seed: ${SEED}"
    echo "==================================================="
    
    # 1. Run Single Conditions
    for COND in "${CONDS[@]}"; do
        FILE="${INPUT_DIR}/isfc_${COND}_loo_${SEED}_desc-zscore.nii.gz"
        
        if [ -f "$FILE" ]; then
            echo "Running ${COND}..."
            python code/TI_code/isfc/run_isfc_behavior_corr.py \
                --input_file "$FILE" \
                --condition "$COND" \
                --csv_file "$CSV" \
                --behavior_col "$COL" \
                --n_perms "$PERMS" \
                --use_tfce
        else
            echo "Warning: File not found: $FILE"
        fi
    done
    
    # 2. Run Contrast: TI1_orig vs TI1_word
    FILE1="${INPUT_DIR}/isfc_TI1_orig_loo_${SEED}_desc-zscore.nii.gz"
    FILE2="${INPUT_DIR}/isfc_TI1_word_loo_${SEED}_desc-zscore.nii.gz"
    
    if [ -f "$FILE1" ] && [ -f "$FILE2" ]; then
        echo "Running Contrast: TI1_orig vs TI1_word..."
        python code/TI_code/isfc/run_isfc_behavior_corr.py \
            --input_file "$FILE1" \
            --contrast_file "$FILE2" \
            --condition "TI1_orig" \
            --contrast_condition "TI1_word" \
            --csv_file "$CSV" \
            --behavior_col "$COL" \
            --n_perms "$PERMS" \
            --use_tfce
    else
        echo "Skipping contrast for ${SEED} (Missing files)"
    fi
     
done

echo "Batch Analysis Complete."
