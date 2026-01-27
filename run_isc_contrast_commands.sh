#!/bin/bash
# Run ISC Contrast Analysis for all 3 pairs, using both LOO and Pairwise methods
# Method: Bootstrap with TFCE

# 1. LOO
echo "--- Running LOO Contrasts ---"
# Note: Using Files found in result/ISC/ISC_bootstrap/fdr/loo/
python shared/run_contrast.py --data_dir /Users/tongshan/Documents/TemporalIntegration/result/ISC/ISC_bootstrap/fdr/loo --cond1 TI1_orig --cond2 TI1_sent --type isc --isc_method loo --method bootstrap --n_perms 1000 --use_tfce
python shared/run_contrast.py --data_dir /Users/tongshan/Documents/TemporalIntegration/result/ISC/ISC_bootstrap/fdr/loo --cond1 TI1_orig --cond2 TI1_word --type isc --isc_method loo --method bootstrap --n_perms 1000 --use_tfce
python shared/run_contrast.py --data_dir /Users/tongshan/Documents/TemporalIntegration/result/ISC/ISC_bootstrap/fdr/loo --cond1 TI1_sent --cond2 TI1_word --type isc --isc_method loo --method bootstrap --n_perms 1000 --use_tfce

# 2. Pairwise
echo "--- Running Pairwise Contrasts ---"
# Note: Using Files found in result/ISC/ISC_bootstrap/fdr/pairwise/
python shared/run_contrast.py --data_dir /Users/tongshan/Documents/TemporalIntegration/result/ISC/ISC_bootstrap/fdr/pairwise --cond1 TI1_orig --cond2 TI1_sent --type isc --isc_method pairwise --method bootstrap --n_perms 1000 --use_tfce
python shared/run_contrast.py --data_dir /Users/tongshan/Documents/TemporalIntegration/result/ISC/ISC_bootstrap/fdr/pairwise --cond1 TI1_orig --cond2 TI1_word --type isc --isc_method pairwise --method bootstrap --n_perms 1000 --use_tfce
python shared/run_contrast.py --data_dir /Users/tongshan/Documents/TemporalIntegration/result/ISC/ISC_bootstrap/fdr/pairwise --cond1 TI1_sent --cond2 TI1_word --type isc --isc_method pairwise --method bootstrap --n_perms 1000 --use_tfce

echo "All contrast analyses complete."
