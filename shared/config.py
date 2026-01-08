"""
Configuration file for Temporal Integration (TI) Analysis Code.
Edit this file to adapt to different environments (e.g., Local vs Sherlock).
"""
import os

# --- PATHS ---
# Default paths (can be overridden by command line arguments in scripts)
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/td/hpf'
OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/TI_code/mask/MNI152_T1_2mm_brain_mask.nii'

# --- ANALYSIS PARAMETERS ---
SUBJECTS = ['11051', '12501', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']
CHUNK_SIZE = 5000
