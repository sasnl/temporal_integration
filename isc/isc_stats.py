import os
import argparse
import numpy as np
import time
import nibabel as nib
from scipy.stats import ttest_1samp
from brainiak.isc import phaseshift_isc
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
import config
from pipeline_utils import load_mask, load_data, save_map, save_plot, run_isc_computation, apply_cluster_threshold, apply_tfce

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
    parser.add_argument('--p_threshold', type=float, default=0.05,
                        help='P-value threshold (default: 0.05)')
    parser.add_argument('--cluster_threshold', type=int, default=0,
                        help='Cluster extent threshold (min voxels). Default: 0 (no threshold)')
    parser.add_argument('--use_tfce', action='store_true',
                        help='Use Threshold-Free Cluster Enhancement (requires permutation/bootstrap). Incompatible with cluster_threshold.')
    parser.add_argument('--tfce_E', type=float, default=0.5,
                        help='TFCE extent parameter (default: 0.5)')
    parser.add_argument('--tfce_H', type=float, default=2.0,
                        help='TFCE height parameter (default: 2.0)')
    
    # Configurable Paths
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to input data (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help=f'Output directory (default: {config.OUTPUT_DIR})')
    parser.add_argument('--mask_file', type=str, default=config.MASK_FILE,
                        help=f'Path to mask file (default: {config.MASK_FILE})')
    parser.add_argument('--chunk_size', type=int, default=config.CHUNK_SIZE,
                        help=f'Chunk size (default: {config.CHUNK_SIZE})')
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

def run_bootstrap_manual(data_4d, n_bootstraps=1000, random_state=42, use_tfce=False, mask_3d=None, tfce_E=0.5, tfce_H=2.0):
    """
    Run bootstrap on 4D map (subjects dimension).
    
    Parameters:
    -----------
    data_4d : array (n_voxels, n_samples)
        Data array
    n_bootstraps : int
        Number of bootstrap iterations
    random_state : int
        Random seed
    use_tfce : bool
        If True, apply TFCE transformation before computing p-values
    mask_3d : 3D array, optional
        Brain mask for TFCE (required if use_tfce=True)
    tfce_E : float
        TFCE extent parameter
    tfce_H : float
        TFCE height parameter
    """
    print(f"Running Bootstrap (n={n_bootstraps})...")
    n_voxels, n_samples = data_4d.shape
    rng = np.random.RandomState(random_state)
    
    # Observed mean
    observed_mean = np.nanmean(data_4d, axis=1)
    
    if use_tfce:
        if mask_3d is None:
            raise ValueError("mask_3d is required when use_tfce=True")
        # Reshape observed mean to 3D for TFCE
        observed_mean_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        observed_mean_3d[mask_3d] = observed_mean
        observed_mean_3d = apply_tfce(observed_mean_3d, mask_3d, E=tfce_E, H=tfce_H, two_sided=True)
        observed_mean = observed_mean_3d[mask_3d]
    
    # Center data per voxel
    # For NaNs: if voxel has NaNs, mean is computed from valid subjects.
    # We center valid subjects by subtracting the observed (nan)mean.
    # NaNs remain NaNs.
    data_centered = data_4d - np.nanmean(data_4d, axis=1, keepdims=True)
    
    null_means = np.zeros((n_voxels, n_bootstraps), dtype=np.float32)
    
    for i in range(n_bootstraps):
        # Resample indices with replacement
        indices = rng.randint(0, n_samples, size=n_samples)
        sample = data_centered[:, indices]
        perm_mean = np.nanmean(sample, axis=1)
        
        if use_tfce:
            # Apply TFCE to permuted map
            perm_mean_3d = np.zeros(mask_3d.shape, dtype=np.float32)
            perm_mean_3d[mask_3d] = perm_mean
            perm_mean_3d = apply_tfce(perm_mean_3d, mask_3d, E=tfce_E, H=tfce_H, two_sided=True)
            perm_mean = perm_mean_3d[mask_3d]
        
        null_means[:, i] = perm_mean
        
    # P-value: Proportion of null means >= observed mean (two-sided: use absolute values)
    with np.errstate(invalid='ignore'):
         p_values = np.sum(np.abs(null_means) >= np.abs(observed_mean[:, np.newaxis]), axis=1) / (n_bootstraps + 1)
    
    return observed_mean, p_values



def run_phaseshift(condition, roi_id, n_perms, data_dir, mask_file, chunk_size=config.CHUNK_SIZE, use_tfce=False, tfce_E=0.5, tfce_H=2.0):
    """
    Run Phase Shift randomization using brainiak's optimized phaseshift_isc function.
    
    Parameters:
    -----------
    use_tfce : bool
        If True, apply TFCE transformation before computing p-values
    tfce_E : float
        TFCE extent parameter
    tfce_H : float
        TFCE height parameter
    """
    print(f"Running Phase Shift (n={n_perms}) using brainiak.isc.phaseshift_isc...")
    
    mask, affine = load_mask(mask_file, roi_id=roi_id)
    if np.sum(mask) == 0: raise ValueError("Empty mask")
    
    group_data = load_data(condition, config.SUBJECTS, mask, data_dir)
    if group_data is None: raise ValueError("No data")
    
    n_trs, n_voxels, n_subs = group_data.shape
    
    # Optimize: Since data is already processed with nan_to_num, no NaNs expected
    # Setting tolerate_nans=False provides significant speedup per documentation
    # Documentation states: "Note that accommodating NaNs may be notably slower"
    has_nans = np.any(np.isnan(group_data))
    tolerate_nans_flag = True if has_nans else False
    if not tolerate_nans_flag:
        print(f"Data contains no NaNs - using tolerate_nans=False for faster computation")
    else:
        print(f"Warning: Data contains NaNs - computation may be slower")
    
    # Use brainiak's optimized phaseshift_isc function
    # This handles phase randomization and ISC computation efficiently
    # Note: In leave-one-out mode, only the left-out subject is phase-randomized per iteration
    # This is more efficient than randomizing all subjects
    print(f"Computing phase-shifted ISC with {n_perms} permutations...")
    print(f"Data shape: {group_data.shape} (TRs × voxels × subjects)")
    print(f"Processing {n_voxels:,} voxels across {n_subs} subjects...")
    print(f"Note: This may take a while for large datasets. Consider using --roi_id for faster testing.")
    
    import time
    start_time = time.time()
    
    # Process in chunks if chunk_size is specified and smaller than total voxels
    # This helps with memory management and may allow better parallelization
    if chunk_size > 0 and chunk_size < n_voxels:
        print(f"Processing {n_voxels:,} voxels in chunks of {chunk_size:,}...")
        n_chunks = int(np.ceil(n_voxels / chunk_size))
        
        # Pre-allocate output arrays
        observed = np.zeros(n_voxels, dtype=np.float32)
        p_values = np.zeros(n_voxels, dtype=np.float32)
        distribution = np.zeros((n_perms, n_voxels), dtype=np.float32)
        
        # Process each chunk
        for chunk_idx in range(n_chunks):
            start_vox = chunk_idx * chunk_size
            end_vox = min((chunk_idx + 1) * chunk_size, n_voxels)
            chunk_voxels = end_vox - start_vox
            
            print(f"  Processing chunk {chunk_idx + 1}/{n_chunks} (voxels {start_vox:,} to {end_vox:,})...")
            chunk_start_time = time.time()
            
            # Extract chunk of voxels: (n_TRs, chunk_voxels, n_subjects)
            group_data_chunk = group_data[:, start_vox:end_vox, :]
            
            # Process chunk with phaseshift_isc
            # Use chunk-specific random state to ensure independent phase randomizations
            # while maintaining reproducibility
            chunk_rng = 42 + chunk_idx if chunk_size > 0 else 42
            obs_chunk, p_chunk, dist_chunk = phaseshift_isc(
                group_data_chunk,
                pairwise=False,
                summary_statistic='median',
                n_shifts=n_perms,
                side='right',
                tolerate_nans=tolerate_nans_flag,
                random_state=chunk_rng
            )
            
            # Store results
            observed[start_vox:end_vox] = obs_chunk
            p_values[start_vox:end_vox] = p_chunk
            distribution[:, start_vox:end_vox] = dist_chunk
            
            chunk_elapsed = time.time() - chunk_start_time
            print(f"    Chunk {chunk_idx + 1} completed in {chunk_elapsed:.2f}s ({chunk_elapsed/60:.2f} min)")
        
        elapsed_time = time.time() - start_time
        print(f"Phase shift computation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    else:
        # Process all voxels at once (original approach)
        print(f"Processing all {n_voxels:,} voxels at once...")
        observed, p_values, distribution = phaseshift_isc(
            group_data,
            pairwise=False,
            summary_statistic='median',
            n_shifts=n_perms,
            side='right',
            tolerate_nans=tolerate_nans_flag,
            random_state=42
        )
        
        elapsed_time = time.time() - start_time
        print(f"Phase shift computation completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # observed shape: (n_voxels,)
    # p_values shape: (n_voxels,)
    # distribution shape: (n_shifts, n_voxels)
    
    obs_mean = observed  # Already the summary statistic (median)
    
    # Handle TFCE if requested
    if use_tfce:
        print("Applying TFCE to observed ISC map...")
        # Apply TFCE to observed map
        obs_mean_3d = np.zeros(mask.shape, dtype=np.float32)
        obs_mean_3d[mask] = obs_mean
        obs_mean_3d = apply_tfce(obs_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
        obs_mean = obs_mean_3d[mask]
        
        # Apply TFCE to null distribution and recompute p-values
        print("Applying TFCE to null distribution...")
        null_distribution_tfce = np.zeros_like(distribution, dtype=np.float32)
        for i in range(n_perms):
            if (i + 1) % 100 == 0:
                print(f"  TFCE permutation {i+1}/{n_perms}")
            null_map_3d = np.zeros(mask.shape, dtype=np.float32)
            null_map_3d[mask] = distribution[i, :]
            null_map_3d = apply_tfce(null_map_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
            null_distribution_tfce[i, :] = null_map_3d[mask]
        
        # Recompute p-values with TFCE-transformed data
        with np.errstate(invalid='ignore'):
            count_greater = np.sum(np.abs(null_distribution_tfce) >= np.abs(obs_mean[:, np.newaxis]), axis=0)
        p_values = (count_greater + 1) / (n_perms + 1)
    
    # Convert back to 3D for return
    obs_mean_3d = np.zeros(mask.shape, dtype=np.float32)
    obs_mean_3d[mask] = obs_mean
    
    p_values_3d = np.ones(mask.shape, dtype=np.float32)
    p_values_3d[mask] = p_values
    
    return obs_mean_3d, p_values_3d, mask, affine 

def main():
    args = parse_args()
    method = args.method
    roi_id = args.roi_id
    threshold = args.p_threshold
    output_dir = args.output_dir
    data_dir = args.data_dir
    mask_file = args.mask_file
    chunk_size = args.chunk_size
    
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
        mean_map, p_values_3d, mask_data, mask_affine = run_phaseshift(
            args.condition, roi_id, args.n_perms, 
            data_dir=data_dir, mask_file=mask_file, chunk_size=chunk_size,
            use_tfce=args.use_tfce, tfce_E=args.tfce_E, tfce_H=args.tfce_H
        )
        
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
            if args.use_tfce:
                print("Warning: TFCE requires permutation/bootstrap. T-test does not support TFCE. Ignoring --use_tfce.")
            mean_vals, p_values = run_ttest(masked_data)
        elif method == 'bootstrap':
            mean_vals, p_values = run_bootstrap_manual(
                masked_data, n_bootstraps=args.n_perms, 
                use_tfce=args.use_tfce, mask_3d=mask_data, 
                tfce_E=args.tfce_E, tfce_H=args.tfce_H
            )
        
        # Reconstruct 3D maps
        mean_map = np.zeros(mask_data.shape, dtype=np.float32)
        mean_map[mask_data] = mean_vals
        
        p_values_3d = np.ones(mask_data.shape, dtype=np.float32)
        p_values_3d[mask_data] = p_values
            
        # Extract filename base
        input_base = os.path.basename(args.input_map).replace('.nii.gz', '').replace('_desc-zscore', '').replace('_desc-raw', '')
        base_name = f"{input_base}_{method}"

    # Results Processing
    roi_suffix = f"_roi{roi_id}" if roi_id is not None else ""
    if roi_suffix not in base_name: # Avoid double suffix if it was in input name
        base_name += roi_suffix
    
    # TFCE suffix
    tfce_suffix = "_tfce" if args.use_tfce else ""
    if tfce_suffix:
        base_name += tfce_suffix
    
    # Check incompatibility
    if args.use_tfce and args.cluster_threshold > 0:
        print("Warning: TFCE and cluster_threshold are incompatible. Ignoring cluster_threshold.")
        args.cluster_threshold = 0
        
    # Save Outputs
    # 1. P-value map
    p_path = os.path.join(output_dir, f"{base_name}_desc-pvalues.nii.gz")
    save_map(p_values_3d, mask_data, mask_affine, p_path)
    
    # 2. Thresholded Map (Significant Only)
    sig_map = mean_map.copy()
    sig_map[p_values_3d >= threshold] = 0
    
    # Apply cluster threshold if requested (and not using TFCE)
    if args.cluster_threshold > 0:
        sig_map = apply_cluster_threshold(sig_map, args.cluster_threshold)
        
    clust_suffix = f"_clust{args.cluster_threshold}" if args.cluster_threshold > 0 else ""
    sig_path = os.path.join(output_dir, f"{base_name}_desc-sig_p{str(threshold).replace('.', '')}{clust_suffix}.nii.gz")
    save_map(sig_map, mask_data, mask_affine, sig_path)
    
    # 3. Plot
    plot_path = os.path.join(output_dir, f"{base_name}_desc-sig.png")
    save_plot(sig_path, plot_path, f"Significant ISC ({method}, p<{threshold})", positive_only=True)
    
    print(f"Stats analysis finished.")
    print(f"Outputs:\n  {p_path}\n  {sig_path}\n  {plot_path}")

if __name__ == "__main__":
    main()
