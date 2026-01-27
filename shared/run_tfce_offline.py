
import os
import argparse
import numpy as np
import nibabel as nib
from joblib import Parallel, delayed
import sys

# Import shared utils
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from pipeline_utils import save_map, apply_tfce, load_mask
import config

def _process_perm_tfce(i, perm_data, mask, E, H, dh):
    """
    Helper for parallel TFCE on permutation maps.
    perm_data: (V,) flattened map of one permutation
    """
    perm_map_3d = np.zeros(mask.shape, dtype=np.float32)
    perm_map_3d[mask] = perm_data
    
    # Apply TFCE
    tfce_map = apply_tfce(perm_map_3d, mask, E=E, H=H, dh=dh, two_sided=False)
    
    # Return Max Statistic for FWER
    return np.max(np.abs(tfce_map))

def run_tfce_offline(observed_map_path, perm_maps_path, output_dir, mask_file, roi_id=None, 
                     E=0.5, H=2.0, dh=0.01, n_jobs=-1):
    
    print(f"--- Offline TFCE Statistics ---")
    print(f"Observed Map: {observed_map_path}")
    print(f"Permutations: {perm_maps_path}")
    print(f"Parameters: E={E}, H={H}, dh={dh}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Data
    print("Loading data...")
    mask, affine = load_mask(mask_file, roi_id=roi_id)
    n_voxels_mask = np.sum(mask)
    
    obs_img = nib.load(observed_map_path)
    obs_data_full = obs_img.get_fdata()
    # Mask observed data
    obs_data = obs_data_full[mask]
    
    perm_img = nib.load(perm_maps_path)
    perm_data_full = perm_img.get_fdata() # (X,Y,Z, N_perms)
    # Mask permutation data -> (V, N_perms)
    perm_data = perm_data_full[mask, :]
    n_perms = perm_data.shape[1]
    
    print(f"Data loaded. Mask voxels: {n_voxels_mask}, Permutations: {n_perms}")
    
    # 2. Compute Observed TFCE
    print("Computing Observed TFCE...")
    obs_tfce_3d = apply_tfce(obs_data_full, mask, E=E, H=H, dh=dh, two_sided=False)
    obs_tfce_flat = obs_tfce_3d[mask]
    
    # 3. Compute Null Distribution (Max TFCE)
    print(f"Computing Null TFCE Distribution ({n_perms} permutations)...")
    
    null_max_stats = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_process_perm_tfce)(
            i, perm_data[:, i], mask, E, H, dh
        ) for i in range(n_perms)
    )
    null_max_stats = np.array(null_max_stats)
    
    # 4. Compute P-values (FWER)
    print("Computing P-values...")
    # Right-tailed test
    # p = (sum(Max_Null >= Obs) + 1) / (B + 1)
    
    # Vectorized calculation
    # For each voxel's Obs score, count how many Null Max scores are >= it
    
    # Sort null max stats for fast searching
    sorted_null_max = np.sort(null_max_stats)
    
    # searchsorted finds the index where value should be inserted to maintain order
    # side='left': indices such that a[i-1] < v <= a[i]
    # We want count of null_max >= v
    # This is len - searchsorted_index (if using side='left' and >= logic carefully)
    # Actually:
    # If null = [1, 2, 3, 4, 5]
    # Obs = 2.5
    # searchsorted(left) -> index 2 (value 3).
    # Count greater = 5 - 2 = 3 (values 3, 4, 5). Correct.
    
    # Obs = 2.0
    # searchsorted(left) -> index 1 (value 2).
    # Count greater = 5 - 1 = 4 (values 2, 3, 4, 5). Correct (since we want >=).
    
    indices = np.searchsorted(sorted_null_max, obs_tfce_flat, side='left')
    count_greater = n_perms - indices
    p_values = (count_greater + 1) / (n_perms + 1)
    
    # 5. Save Results
    print("Saving results...")
    
    # Prepare base name
    base_name = os.path.basename(observed_map_path).replace('.nii.gz', '').replace('_desc-stat', '').replace('_desc-zscore', '')
    # Add tfce suffix if not present
    if 'tfce' not in base_name:
        base_name += f"_tfce_dh{dh}"
    else:
        base_name += f"_dh{dh}"
        
    # Save TFCE Score Map
    stat_path = os.path.join(output_dir, f"{base_name}_desc-tfce.nii.gz")
    save_map(obs_tfce_3d, mask, affine, stat_path)
    
    # Save 1-p Map (for visualization convenience sometimes, typically we save p)
    # Standard P-value map
    p_path = os.path.join(output_dir, f"{base_name}_desc-pvalues.nii.gz")
    p_values_3d = np.ones(mask.shape, dtype=np.float32)
    p_values_3d[mask] = p_values
    save_map(p_values_3d, mask, affine, p_path)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TFCE Offline using saved permutations')
    parser.add_argument('--observed_map', required=True, help='Path to observed statistic map (e.g. desc-stat or raw Z)')
    parser.add_argument('--perm_maps', required=True, help='Path to 4D permutation file (desc-perms)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--mask_file', default=config.MASK_FILE, help='Mask file path')
    parser.add_argument('--roi_id', type=int, default=None, help='ROI ID (optional)')
    parser.add_argument('--tfce_E', type=float, default=0.5, help='TFCE E parameter (default: 0.5)')
    parser.add_argument('--tfce_H', type=float, default=2.0, help='TFCE H parameter (default: 2.0)')
    parser.add_argument('--tfce_dh', type=float, default=0.01, help='TFCE step size (default: 0.01)')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    run_tfce_offline(
        args.observed_map, args.perm_maps, args.output_dir, args.mask_file, 
        roi_id=args.roi_id, E=args.tfce_E, H=args.tfce_H, dh=args.tfce_dh, n_jobs=args.n_jobs
    )
