"""
Configuration file for Temporal Integration (TI) Analysis Code.
Edit this file to adapt to different environments (e.g., Local vs Sherlock).
"""
import os

# --- PATHS ---

# Default paths (can be overridden by command line arguments in scripts)
DATA_DIR = '/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td'
OUTPUT_DIR = '/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/isc_analysis_1000_permutations_hpc_0.05'
MASK_FILE = '/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/mask/MNI152_T1_2mm_brain_mask.nii'

# --- ANALYSIS PARAMETERS ---
# SUBJECTS = ['11051', '12501', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']
SUBJECTS= ['12501','12502','12503','12504','12505','12506','12515','12516','12517','12527','12530','12531','12532','12538','12541','12542','12545','12548','11012','11036','11051','11054','9409']
# SUBJECTS = ['11051','12501','12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']
CHUNK_SIZE = 300000
