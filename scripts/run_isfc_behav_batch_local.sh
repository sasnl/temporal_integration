#!/bin/bash

# Batch script to run ISFC-Behavior Correlation for all 3 seeds and conditions
# Seeds:
# 1. Left HG: seed-63_-42_9_r5
# 2. PMC: seed0_-53_2_r5
# 3. Right HG: seed57_-31_5_r5

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

SEEDS=("seed-63_-42_9_r5" "seed0_-53_2_r5" "seed57_-31_5_r5")
CONDS=("TI1_orig" "TI1_sent" "TI1_word")
CSV="data/demographic/60483ASDSpeakerListe-TISubjectDemographic_DATA_2026-01-29_1113_merged.csv"
COL="fsid0079wasi_ii_0007_ffid002642_fsiq"
PERMS=1000
INPUT_DIR="result/ISFC/bootstrap/tfce/loo"

echo "Starting Batch ISFC Correlation Analysis (n=${PERMS})..."

for SEED in "${SEEDS[@]}"; do
    echo "==================================================="
    echo "Processing Seed: ${SEED}"
    echo "==================================================="
    
    # Map Seed to Subfolder (Friendly Name)
    case "$SEED" in
        "seed-63_-42_9_r5") SUBDIR="LpSTS" ;;
        "seed0_-53_2_r5")   SUBDIR="PMC" ;;
        "seed57_-31_5_r5")  SUBDIR="RpSTS" ;;
        *) echo "Unknown Seed: $SEED"; exit 1 ;;
    esac
    
    # 1. Run Single Conditions
    for COND in "${CONDS[@]}"; do
        FILE="${INPUT_DIR}/${SUBDIR}/isfc_${COND}_loo_${SEED}_desc-zscore.nii.gz"
        
        if [ -f "$FILE" ]; then
            echo "Running ${COND}..."
            python isfc/run_isfc_behavior_corr.py \
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
    FILE1="${INPUT_DIR}/${SUBDIR}/isfc_TI1_orig_loo_${SEED}_desc-zscore.nii.gz"
    FILE2="${INPUT_DIR}/${SUBDIR}/isfc_TI1_word_loo_${SEED}_desc-zscore.nii.gz"
    
    if [ -f "$FILE1" ] && [ -f "$FILE2" ]; then
        echo "Running Contrast: TI1_orig vs TI1_word..."
        python isfc/run_isfc_behavior_corr.py \
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
