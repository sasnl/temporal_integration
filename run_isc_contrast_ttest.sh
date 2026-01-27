#!/bin/bash
# Run Uncorrected T-test ISC Contrast Analysis for all 3 pairs, using both LOO and Pairwise methods

DATA_DIR="/Users/tongshan/Documents/TemporalIntegration/result/ISC/ISC_bootstrap/fdr"

# 1. LOO
echo "--- Running LOO T-tests ---"
python shared/run_contrast.py --data_dir $DATA_DIR/loo --cond1 TI1_orig --cond2 TI1_sent --type isc --isc_method loo --method ttest
python shared/run_contrast.py --data_dir $DATA_DIR/loo --cond1 TI1_orig --cond2 TI1_word --type isc --isc_method loo --method ttest
python shared/run_contrast.py --data_dir $DATA_DIR/loo --cond1 TI1_sent --cond2 TI1_word --type isc --isc_method loo --method ttest

# 2. Pairwise
echo "--- Running Pairwise T-tests ---"
python shared/run_contrast.py --data_dir $DATA_DIR/pairwise --cond1 TI1_orig --cond2 TI1_sent --type isc --isc_method pairwise --method ttest
python shared/run_contrast.py --data_dir $DATA_DIR/pairwise --cond1 TI1_orig --cond2 TI1_word --type isc --isc_method pairwise --method ttest
python shared/run_contrast.py --data_dir $DATA_DIR/pairwise --cond1 TI1_sent --cond2 TI1_word --type isc --isc_method pairwise --method ttest

echo "All uncorrected T-test analyses complete."
