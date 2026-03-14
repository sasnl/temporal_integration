#!/bin/bash
# Script to run ISFC Bootstrap TFCE Pipeline for 4 new ROI seeds (NT group)
# Seeds: Left aSTS, Right aSTS, Left HG, Right HG

# Directories
TI_CODE="/Users/tongshan/Documents/TemporalIntegration/code/TI_code"
MASK_DIR="$TI_CODE/mask"
DATA_DIR="/Users/tongshan/Documents/TemporalIntegration/data/nt_ti_processed"
RESULT_DIR="/Users/tongshan/Documents/TemporalIntegration/result/NT/ISFC/bootstrap"

# Seed files
LEFT_ASTS="$MASK_DIR/08-4mm_Left_aSTS_4mm_-56_-16_-6_-56_-16_-6.nii"
RIGHT_ASTS="$MASK_DIR/16-4mm_Right_aSTS_4mm_-56_-16_-6_56_-16_-6.nii"
LEFT_HG="$MASK_DIR/HO_Left_Heschls_Gyrus.nii"
RIGHT_HG="$MASK_DIR/HO_Right_Heschls_Gyrus.nii"

# Create Output Directory
mkdir -p "$RESULT_DIR/tfce/loo"

cd "$TI_CODE"

echo "=== Starting ISFC Analysis (New Seeds, Bootstrap & TFCE) ==="
echo "Pipeline started at: $(date)"

N_PERMS=1000
P_THRESH=0.05

# --- Left aSTS ---
echo "=== Left aSTS ==="
python isfc/run_isfc_pipeline.py --condition TI1_orig --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$LEFT_ASTS" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"
python isfc/run_isfc_pipeline.py --condition TI1_sent --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$LEFT_ASTS" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"
python isfc/run_isfc_pipeline.py --condition TI1_word --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$LEFT_ASTS" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"

# --- Right aSTS ---
echo "=== Right aSTS ==="
python isfc/run_isfc_pipeline.py --condition TI1_orig --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$RIGHT_ASTS" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"
python isfc/run_isfc_pipeline.py --condition TI1_sent --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$RIGHT_ASTS" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"
python isfc/run_isfc_pipeline.py --condition TI1_word --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$RIGHT_ASTS" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"

# --- Left Heschl's Gyrus ---
echo "=== Left Heschl's Gyrus ==="
python isfc/run_isfc_pipeline.py --condition TI1_orig --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$LEFT_HG" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"
python isfc/run_isfc_pipeline.py --condition TI1_sent --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$LEFT_HG" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"
python isfc/run_isfc_pipeline.py --condition TI1_word --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$LEFT_HG" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"

# --- Right Heschl's Gyrus ---
echo "=== Right Heschl's Gyrus ==="
python isfc/run_isfc_pipeline.py --condition TI1_orig --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$RIGHT_HG" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"
python isfc/run_isfc_pipeline.py --condition TI1_sent --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$RIGHT_HG" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"
python isfc/run_isfc_pipeline.py --condition TI1_word --isfc_method loo --stats_method bootstrap --n_perms $N_PERMS --use_tfce --tfce_dh 0.01 --seed_file "$RIGHT_HG" --p_threshold $P_THRESH --data_dir "$DATA_DIR" --output_dir "$RESULT_DIR/tfce/loo"

echo "=== Analysis Complete. Starting Re-thresholding ==="

python shared/rethreshold_results.py --result_dir "$RESULT_DIR/tfce" --thresholds 0.01 0.005 0.001 --overwrite

echo "Done at: $(date)"
