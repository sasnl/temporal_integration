import os
import argparse
import numpy as np
import time
import nibabel as nib
from scipy.stats import ttest_1samp
from brainiak.isc import phaseshift_isc
from joblib import Parallel, delayed
from isc_utils import load_mask, load_data, save_map, save_plot
import config

def parse_args():
    parser = argparse.ArgumentParser(description='Step 2: Statistical Analysis for ISC')
    parser.add_argument('--input_map', type=str, 
                        help='Path to 4D ISC map (Z-score recommended for T-test/Bootstrap). Required for T-test/Bootstrap.')
    parser.add_argument('--method', type=str, choices=['ttest', 'bootstrap', 'phaseshift'], required=True,
                        help='Statistical method: "ttest", "bootstrap", or "phaseshift"')
    parser.add_argument('--condition', type=str, 
                        help='Condition name (e.g., TI1_orig). Required for Phase Shift.')
    parser.add_argument('--n_perms', type=int, default=1000,
                        help='Number of permutations/bootstraps (default: 1000)')
    parser.add_argument('--roi_id', type=int, default=None,
                        help='Optional: ROI ID to mask (default: Whole Brain)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='P-value threshold (default: 0.05)')
    
    # Configurable Paths
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to input data (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help=f'Output directory (default: {config.OUTPUT_DIR})')
    parser.add_argument('--mask_file', type=str, default=config.MASK_FILE,
                        help=f'Path to mask file (default: {config.MASK_FILE})')
    return parser.parse_args()

def run_ttest(data_4d):
    """
    Run one-sample T-test against 0 on the last dimension of data (subjects/pairs).
    data_4d: (n_voxels, n_samples)
    """
    print("Running T-test...")
    # axis -1 is the sample dimension
    t_stats, p_values = ttest_1samp(data_4d, popmean=0, axis=-1, nan_policy='omit')
    mean_map = np.nanmean(data_4d, axis=-1)
    return mean_map, p_values

def run_bootstrap_manual(data_4d, n_bootstraps=1000, random_state=42):
    """
    Run bootstrap on 4D map (subjects dimension).
    """
    print(f"Running Bootstrap (n={n_bootstraps})...")
    n_voxels, n_samples = data_4d.shape
    rng = np.random.RandomState(random_state)
    
    # Observed mean
    observed_mean = np.nanmean(data_4d, axis=1)
    
    # Center data per voxel
    # For NaNs: if voxel has NaNs, mean is computed from valid subjects.
    # We center valid subjects by subtracting the observed (nan)mean.
    # NaNs remain NaNs.
    data_centered = data_4d - observed_mean[:, np.newaxis]
    
    null_means = np.zeros((n_voxels, n_bootstraps), dtype=np.float32)
    
    for i in range(n_bootstraps):
        # Resample indices with replacement
        indices = rng.randint(0, n_samples, size=n_samples)
        sample = data_centered[:, indices]
        null_means[:, i] = np.nanmean(sample, axis=1)
        
    # P-value: Proportion of null means >= observed mean
    # NaNs in null_means or observed_mean will result in False for comparison, or warning.
    # We should handle this validly.
    with np.errstate(invalid='ignore'):
         p_values = np.sum(null_means >= observed_mean[:, np.newaxis], axis=1) / n_bootstraps
    
    # Two-sided correction if needed, but usually ISC is one-sided (>0). 
    # If we want two-sided: p = 2 * min(p_one_sided, 1 - p_one_sided)
    # Let's stick to one-sided > 0 for standard ISC.
    
    return observed_mean, p_values

def process_phaseshift_chunk(chunk_data, n_perms):
    # phaseshift_isc returns: observed, p, distribution
    # pairwise=False (LOO) is standard for group
    observed, p, _ = phaseshift_isc(chunk_data, pairwise=False, n_shifts=n_perms, random_state=42)
    return observed, p

def run_phaseshift(condition, roi_id, n_perms, data_dir, mask_file):
    """
    Run Phase Shift randomization. Requires plain raw data (no Z-score maps).
    """
    print("Running Phase Shift (requires reloading raw data)...")
    
    # Load Mask
    mask, _ = load_mask(mask_file, roi_id=roi_id)
    if np.sum(mask) == 0:
        raise ValueError("Empty mask.")

    # Load Data
    group_data = load_data(condition, config.SUBJECTS, mask, data_dir)
    if group_data is None:
        raise ValueError("No data loaded for phase shift.")

    n_trs, n_voxels, n_subs = group_data.shape
    
    # Run in Chunks
    n_chunks = int(np.ceil(n_voxels / config.CHUNK_SIZE))
    chunks = []
    for i in range(n_chunks):
        start_idx = i * config.CHUNK_SIZE
        end_idx = min((i + 1) * config.CHUNK_SIZE, n_voxels)
        chunks.append(group_data[:, start_idx:end_idx, :])

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_phaseshift_chunk)(chunk, n_perms) for chunk in chunks
    )
    
    # Reassemble
    mean_map = np.zeros(n_voxels, dtype=np.float32)
    p_value_map = np.zeros(n_voxels, dtype=np.float32)
    
    for i, (observed, p) in enumerate(results):
        start_idx = i * config.CHUNK_SIZE
        end_idx = min((i + 1) * config.CHUNK_SIZE, n_voxels)
        mean_map[start_idx:end_idx] = observed
        p_value_map[start_idx:end_idx] = p
        
    return mean_map, p_value_map, mask

def main():
    args = parse_args()
    method = args.method
    roi_id = args.roi_id
    threshold = args.threshold
    output_dir = args.output_dir
    data_dir = args.data_dir
    mask_file = args.mask_file
    
    print(f"--- Step 2: ISC Statistics ---")
    print(f"Method: {method}")
    print(f"Threshold: {threshold}")
    print(f"Output Dir: {output_dir}")
    print(f"Data Dir: {data_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    mask_affine = None
    mask_data = None
    
    # Logic Branch
    if method == 'phaseshift':
        if not args.condition:
            print("Error: --condition is required for phaseshift.")
            return
        # Phase shift loads its own data/mask inside the function to ensure compatibility
        mean_map, p_values, mask_data = run_phaseshift(args.condition, roi_id, args.n_perms, data_dir=data_dir, mask_file=mask_file)
        # Need to load mask affine separately if not returned
        _, mask_affine = load_mask(mask_file, roi_id=roi_id)
        
        # Base name for output
        base_name = f"isc_{args.condition}_{method}"
        
    else:
        # Map-based methods
        if not args.input_map:
            print("Error: --input_map is required for ttest/bootstrap.")
            return
            
        print(f"Loading input map: {args.input_map}")
        img = nib.load(args.input_map)
        data_4d = img.get_fdata(dtype=np.float32) # (X, Y, Z, N)
        mask_affine = img.affine
        
        # Convert to 2D (voxels, subjects) using mask
        # We need the mask to extract voxels. 
        # If input_map is full brain volume, we need to apply mask.
        mask_data, _ = load_mask(mask_file, roi_id=roi_id)
        
        # Check shapes
        if data_4d.shape[:3] != mask_data.shape:
             print("Error: Input map dimensions do not match mask.")
             return
             
        # Extract voxels
        # data_4d[mask] -> returns (n_voxels, n_samples)
        # Note: numpy boolean indexing on 4D array:
        # If mask is 3D, data_4d[mask] selects elements. 
        # Wait, data_4d[mask] will flatten the spatial dims?
        # Yes, data_4d[mask] returns (n_voxels, n_samples)
        
        masked_data = data_4d[mask_data] # Result shape: (n_voxels_in_mask, n_samples)
        
        if method == 'ttest':
            mean_map, p_values = run_ttest(masked_data)
        elif method == 'bootstrap':
            mean_map, p_values = run_bootstrap_manual(masked_data, n_bootstraps=args.n_perms)
            
        # Extract filename base
        input_base = os.path.basename(args.input_map).replace('.nii.gz', '').replace('_desc-zscore', '').replace('_desc-raw', '')
        base_name = f"{input_base}_{method}"

    # Results Processing
    roi_suffix = f"_roi{roi_id}" if roi_id is not None else ""
    if roi_suffix not in base_name: # Avoid double suffix if it was in input name
        base_name += roi_suffix
        
    # Save Outputs
    # 1. P-value map
    p_path = os.path.join(output_dir, f"{base_name}_desc-pvalues.nii.gz")
    save_map(p_values, mask_data, mask_affine, p_path)
    
    # 2. Thresholded Map (Significant Only)
    sig_indices = p_values < threshold
    sig_map = np.zeros_like(mean_map)
    sig_map[sig_indices] = mean_map[sig_indices]
    
    sig_path = os.path.join(output_dir, f"{base_name}_desc-sig_p{str(threshold).replace('.', '')}.nii.gz")
    save_map(sig_map, mask_data, mask_affine, sig_path)
    
    # 3. Plot
    plot_path = os.path.join(output_dir, f"{base_name}_desc-sig.png")
    save_plot(sig_path, plot_path, f"Significant ISC ({method}, p<{threshold})")
    
    print(f"Stats analysis finished.")
    print(f"Outputs:\n  {p_path}\n  {sig_path}")

if __name__ == "__main__":
    main()
