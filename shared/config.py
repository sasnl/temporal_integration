"""
Configuration file for Temporal Integration (TI) Analysis Code.
Edit this file to adapt to different environments (e.g., Local vs Sherlock).
"""
import os

# --- PATHS ---
# Default paths (can be overridden by command line arguments in scripts)
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/ti_processed'
OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/TI_code/mask/MNI152_T1_2mm_brain_mask.nii'

# --- ANALYSIS PARAMETERS ---
SUBJECT_LISTS = {
    'TI1_orig': [
        '12501', '12502', '12503', '12505', '12506',
        '12515', '12516', '12517', '12527', '12530',
        '12531', '12532', '12538', '12542', '12545',
        '12548', '11012', '11036', '11051', '11054',
        '9409'
    ],
    'TI1_sent': [
        '12501', '12502', '12503', '12504', '12505',
        '12506', '12515', '12516', '12517', '12527',
        '12530', '12531', '12532', '12538', '12541',
        '12542', '12545', '12548', '11036', '11051',
        '11054', '9409'
    ],
    'TI1_word': [
        '12501', '12503', '12504', '12505', '12506',
        '12515', '12516', '12517', '12527', '12530',
        '12531', '12532', '12538', '12542', '12545',
        '12548', '11036', '11051', '11054', '9409'
    ]
}

# Default fall-back (union of all or just generic) if needed, but scripts should use the dict above
SUBJECTS = SUBJECT_LISTS['TI1_orig'] # Backward compatibility default
CHUNK_SIZE = 5000

# --- ASD GROUP ---
ASD_DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/asd_ti_processed'
ASD_OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result/ASD'
GROUP_OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result/group_comparison'

ASD_SUBJECT_LISTS = {
    'TI1_orig': [
        '12511', '12520', '12521', '12529', '12539', '12544',
        '12546', '12547', '12549', '14999', '9143', '9277',
        '9317', '9488'
    ],
    'TI1_sent': [
        '12511', '12520', '12521', '12525', '12529', '12539',
        '12544', '12546', '12547', '12549', '9143', '9277',
        '9317'
    ],
    'TI1_word': [
        '12511', '12529', '12539', '12540', '12544', '12546',
        '12547', '12549', '14999', '9143', '9277', '9317',
        '9488'
    ]
}
