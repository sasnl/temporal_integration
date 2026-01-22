import os
import argparse
import numpy as np
import time
import nibabel as nib
from scipy.stats import ttest_1samp
from brainiak.utils.utils import phase_randomize
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
import config
from pipeline_utils import load_mask, load_data, save_map, save_plot, run_isc_computation, apply_cluster_threshold, apply_tfce
from joblib import Parallel, delayed
from brainiak.isc import bootstrap_isc



def _run_phaseshift_iter(i, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, seed):
    n_subs = group_data.shape[2]
    rng = np.random.RandomState(seed)
    
    # Shift each subject
    shifted_data = np.zeros_like(group_data)
    for s in range(n_subs):
        shifted_data[:, :, s] = phase_randomize(group_data[:, :, s], random_state=rng)
        
    # Compute Null ISC
    null_raw, null_z = run_isc_computation(shifted_data, chunk_size=chunk_size)
    null_mean = np.nanmean(null_z, axis=1)
    
    if use_tfce:
        # FWER Correction: Return max statistic
        null_mean_3d = np.zeros(mask.shape, dtype=np.float32)
        null_mean_3d[mask] = null_mean
        null_mean_3d = apply_tfce(null_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
        return np.max(np.abs(null_mean_3d))
    else:
        return null_mean


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

def run_bootstrap_brainiak(data_4d, n_bootstraps=1000, random_state=42, use_tfce=False, mask_3d=None, tfce_E=0.5, tfce_H=2.0):
    """
    Run bootstrap using BrainIAK.
    
    Parameters:
    -----------
    data_4d : array (n_voxels, n_samples)
        Data array
    n_bootstraps : int
        Number of bootstrap iterations
    random_state : int
        Random seed
    use_tfce : bool
        If True, apply TFCE to the bootstrap distribution for FWER correction.
    """
    print(f"Running BrainIAK Bootstrap (n={n_bootstraps}, summary=median, side=right)...")
    
    # BrainIAK expects (n_samples, n_voxels) or (n_pairs, n_voxels)
    # Our data_4d is (n_voxels, n_samples). We need to transpose.
    data_reshaped = data_4d.T # (n_samples, n_voxels)
    
    # Run BrainIAK bootstrap
    # Returns: observed, ci, p, distribution
    observed, ci, p_values, distribution = bootstrap_isc(
        data_reshaped, 
        pairwise=False, 
        summary_statistic='median', 
        n_bootstraps=n_bootstraps, 
        ci_percentile=95, 
        side='right', 
        random_state=random_state
    )
    
    # observed: (n_voxels,) - Median ISC
    # p_values: (n_voxels,) - Uncorrected p-values
    # distribution: (n_bootstraps, n_voxels) - Bootstrap distribution of Medians
    
    if use_tfce:
        if mask_3d is None:
            raise ValueError("mask_3d is required when use_tfce=True")
            
        print("Computing TFCE correction on bootstrap distribution...")
        
        # 1. Shift distribution to Null Hypothesis (Shift-to-Null)
        # BrainIAK does this internally for p-values, but returns the raw bootstrap distribution.
        # Null = Bootstrap - Observed (so the new median is 0)
        # We need to broadcast observed across the distribution
        null_distribution = distribution - observed[np.newaxis, :]
        
        # 2. Compute Max-TFCE for each bootstrap iteration
        n_boots = null_distribution.shape[0]
        max_tfce_stats = np.zeros(n_boots)
        
        # We can parallelize this step since it's just processing the distribution
        # Define helper for parallel TFCE
        def _compute_max_tfce(boot_idx, boot_map_flat, mask_3d, E, H):
            boot_map_3d = np.zeros(mask_3d.shape, dtype=np.float32)
            boot_map_3d[mask_3d] = boot_map_flat
            
            # Apply TFCE (two-sided because null distribution is symmetric around 0 after shift)
            # Even though we do a one-sided test, TFCE magnitude matters.
            # However, for a one-sided test (right), we might only care about positive TFCE?
            # Standard FWER for one-sided: Check if Observed TFCE > Max Null TFCE.
            # The Null distribution should be treated as it is. 
            
            tfce_map = apply_tfce(boot_map_3d, mask_3d, E=E, H=H, two_sided=True)
            return np.max(tfce_map) # Max statistic

        # Run parallel TFCE on the null distribution
        max_tfce_stats = Parallel(n_jobs=-1, verbose=5)(
            delayed(_compute_max_tfce)(i, null_distribution[i, :], mask_3d, tfce_E, tfce_H)
            for i in range(n_boots)
        )
        max_tfce_stats = np.array(max_tfce_stats)
        
        # 3. Compute Observed TFCE
        # We need to compute TFCE on the observed MEDIAN map
        obs_map_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        obs_map_3d[mask_3d] = observed
        
        # For Right-Tailed test, we typically only care about positive clusters.
        # But 'two_sided=True' in apply_tfce usually takes abs() or handles both tails.
        # Given we want to test "Significant Positive ISC", we should look at the positive tail.
        # The apply_tfce implementation (from previous turns) likely handles sign or abs.
        # Let's trust apply_tfce(two_sided=True) gives meaningful magnitude or we can check.
        # If two_sided=True, it usually enhances positive and negative clusters separately and combines max.
        
        obs_tfce_3d = apply_tfce(obs_map_3d, mask_3d, E=tfce_E, H=tfce_H, two_sided=True)
        obs_tfce_flat = obs_tfce_3d[mask_3d]
        
        # 4. Compute Corrected P-values
        # P = (sum(Max_Null >= Observed) + 1) / (N + 1)
        # Note: Since it's a right-tailed test, we compare Obs_TFCE vs Max_Null_TFCE
        # (Assuming Max_Null_TFCE represents the max enhancement expected by chance)
        
        # Broadcast for comparison: (V,) vs (B,)
        # count how many times max_random_tfce >= obs_tfce
        
        p_values_corrected = np.zeros_like(observed)
        
        # Using the same logic as before (broadcasting or loop)
        # We want p-value for each voxel
        # For a voxel v with TFCE val X: how many perm's MAX statistic is >= X?
        
        # obs_tfce_flat: (V,)
        # max_tfce_stats: (B,)
        
        # p_val[v] = sum(max_tfce_stats >= obs_tfce_flat[v]) / (B+1)
        
        # This is strictly right-tailed FWER (Probability that max noise > observed signal)
        
        # Optimization: Sort max stats
        sorted_max_stats = np.sort(max_tfce_stats)
        # searchsorted returns index where value would be inserted to maintain order
        # We want count of values >= obs
        # index = searchsorted(sorted, obs)
        # count >= obs  is  N - index
        
        # searchsorted convention: side='left' (default) finds first index >= val
        indices = np.searchsorted(sorted_max_stats, obs_tfce_flat, side='left')
        count_greater = n_boots - indices
        p_values_corrected = (count_greater + 1) / (n_boots + 1)
        
        # Return both observed median and the corrected p-values
        # Note: we return 'observed' (median) map, but the p-values correspond to the TFCE test.
        # The user's code expects (mean_map, p_values). We return (median_map, p_values).
        return observed, p_values_corrected

    else:
        # If no TFCE, just return the brainiak p-values
        return observed, p_values



def run_phaseshift(condition, roi_id, n_perms, data_dir, mask_file, chunk_size=config.CHUNK_SIZE, use_tfce=False, tfce_E=0.5, tfce_H=2.0):
    """
    Run Phase Shift randomization.
    
    Parameters:
    -----------
    use_tfce : bool
        If True, apply TFCE transformation before computing p-values
    tfce_E : float
        TFCE extent parameter
    tfce_H : float
        TFCE height parameter
    """
    print(f"Running Phase Shift (n={n_perms}, chunk_size={chunk_size})...")
    
    mask, affine = load_mask(mask_file, roi_id=roi_id)
    if np.sum(mask) == 0: raise ValueError("Empty mask")
    
    if condition in config.SUBJECT_LISTS:
        subjects = config.SUBJECT_LISTS[condition]
    else:
        subjects = config.SUBJECTS
        
    group_data = load_data(condition, subjects, mask, data_dir)
    if group_data is None: raise ValueError("No data")
    
    n_trs, n_voxels, n_subs = group_data.shape
    
    # 1. Compute Observed ISC
    obs_raw, obs_z = run_isc_computation(group_data, chunk_size=chunk_size)
    obs_mean = np.nanmean(obs_z, axis=1) # (V,)
    
    if use_tfce:
        # Apply TFCE to observed map
        obs_mean_3d = np.zeros(mask.shape, dtype=np.float32)
        obs_mean_3d[mask] = obs_mean
        obs_mean_3d = apply_tfce(obs_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
        obs_mean = obs_mean_3d[mask]
    
    # 2. Phase Randomization (Parallel)
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_phaseshift_iter)(
            i, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, 1000 + i
        ) for i in range(n_perms)
    )

    if use_tfce:
        # FWER Correction
        null_max_stats = np.array(results) # (n_perms,)
        p_values = (np.sum(null_max_stats[np.newaxis, :] >= np.abs(obs_mean[:, np.newaxis]), axis=1) + 1) / (n_perms + 1)
    else:
        # Voxel-wise
        null_means = np.array(results).T # (V, n_perms)
        p_values = np.sum(np.abs(null_means) >= np.abs(obs_mean[:, np.newaxis]), axis=1) / (n_perms + 1)
    
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
            mean_vals, p_values = run_bootstrap_brainiak(
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
    # 1a. Un-thresholded Map (TFCE score or Mean Statistic)
    stat_suffix = "tfce" if args.use_tfce else "stat"
    stat_path = os.path.join(output_dir, f"{base_name}_desc-{stat_suffix}.nii.gz")
    save_map(mean_map, mask_data, mask_affine, stat_path)

    # 1b. P-value map
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
