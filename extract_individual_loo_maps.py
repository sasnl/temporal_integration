"""
Extract individual participant LOO ISC/ISFC Z-scored maps from 4D NIfTI files.
Outputs one 3D .nii.gz per subject, organized for Dan's grant.
"""

import os
import nibabel as nib
import numpy as np

# --- Configuration ---
RESULT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'
OUTPUT_BASE = os.path.join(RESULT_DIR, 'for_dan_grant', 'individual_loo_maps')

CONDITIONS = ['TI1_orig', 'TI1_sent', 'TI1_word']

# Subject lists per group per condition (from config.py)
SUBJECT_LISTS = {
    'TD': {
        'TI1_orig': [
            '12501', '12502', '12503', '12505', '12506',
            '12515', '12516', '12517', '12527', '12530',
            '12531', '12532', '12538', '12542', '12545',
            '12548', '11012', '11036', '11051', '11054', '9409'
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
    },
    'ASD': {
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
}

# Map group labels to result directory names
GROUP_DIR = {'TD': 'NT', 'ASD': 'ASD'}

SEEDS = {
    'PMC': 'seed0_-53_2_r5',
    'LpSTS': 'seed-63_-42_9_r5',
    'RpSTS': 'seed57_-31_5_r5',
}


def extract_maps(input_4d_path, subjects, output_dir, prefix):
    """Split a 4D NIfTI into individual 3D maps, one per subject."""
    img = nib.load(input_4d_path)
    data = img.get_fdata()

    n_vols = data.shape[3]
    assert n_vols == len(subjects), (
        f"Mismatch: {input_4d_path} has {n_vols} volumes but expected {len(subjects)} subjects"
    )

    os.makedirs(output_dir, exist_ok=True)
    for i, sub_id in enumerate(subjects):
        sub_data = data[..., i]
        sub_img = nib.Nifti1Image(sub_data, img.affine, img.header)
        out_path = os.path.join(output_dir, f'{prefix}_sub-{sub_id}_desc-zscore.nii.gz')
        nib.save(sub_img, out_path)

    print(f"  Saved {len(subjects)} maps to {output_dir}")


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # --- ISC ---
    print("=== Extracting ISC LOO maps ===")
    for group, group_dir_name in GROUP_DIR.items():
        for cond in CONDITIONS:
            subjects = SUBJECT_LISTS[group][cond]
            input_path = os.path.join(
                RESULT_DIR, group_dir_name, 'ISC', 'bootstrap', 'tfce', 'loo',
                f'isc_{cond}_loo_desc-zscore.nii.gz'
            )
            output_dir = os.path.join(OUTPUT_BASE, 'ISC', group, cond)
            print(f"[{group} / {cond}] {len(subjects)} subjects")
            extract_maps(input_path, subjects, output_dir, f'isc_{cond}_loo')

    # --- ISFC ---
    print("\n=== Extracting ISFC LOO maps ===")
    for group, group_dir_name in GROUP_DIR.items():
        for cond in CONDITIONS:
            subjects = SUBJECT_LISTS[group][cond]
            for seed_name, seed_suffix in SEEDS.items():
                input_path = os.path.join(
                    RESULT_DIR, group_dir_name, 'ISFC', 'bootstrap', 'tfce', 'loo',
                    seed_name,
                    f'isfc_{cond}_loo_{seed_suffix}_desc-zscore.nii.gz'
                )
                output_dir = os.path.join(OUTPUT_BASE, 'ISFC', seed_name, group, cond)
                print(f"[{group} / {cond} / {seed_name}] {len(subjects)} subjects")
                extract_maps(input_path, subjects, output_dir, f'isfc_{cond}_loo_{seed_suffix}')

    print(f"\nDone! All individual maps saved to:\n  {OUTPUT_BASE}")


if __name__ == '__main__':
    main()
