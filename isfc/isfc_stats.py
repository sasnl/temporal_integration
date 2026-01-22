import os
import argparse
import numpy as np
import nibabel as nib
from scipy.stats import ttest_1samp
from brainiak.utils.utils import phase_randomize
from joblib import Parallel, delayed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from pipeline_utils import load_mask, load_data, save_map, save_plot, get_seed_mask, load_seed_data, apply_cluster_threshold, apply_tfce
from isfc_compute import run_isfc_computation 
# Import run_isfc_computation to reuse logic for phase shift re-computation
import config

def _run_bootstrap_iter(i, n_samples, data_centered, use_tfce, mask_3d, tfce_E, tfce_H, seed):
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n_samples, size=n_samples)
    sample = data_centered[:, indices]
    perm_mean = np.nanmean(sample, axis=1)
    
    if use_tfce:
        # Apply TFCE to permuted map
        perm_mean_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        perm_mean_3d[mask_3d] = perm_mean
        perm_mean_3d = apply_tfce(perm_mean_3d, mask_3d, E=tfce_E, H=tfce_H, two_sided=True)
        # Return Max-Statistic for FWER correction
        return np.max(np.abs(perm_mean_3d))
    else:
        return perm_mean

def _run_phaseshift_iter(i, obs_seed_ts, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, seed):
    # Generate surrogate seed by phase randomizing the OBSERVED seed
    surr_seed_ts = phase_randomize(obs_seed_ts, voxelwise=False, random_state=seed)
    
    surr_raw, surr_z = run_isfc_computation(group_data, surr_seed_ts, pairwise=False, chunk_size=chunk_size)
    null_mean = np.nanmean(surr_z, axis=1) # Mean over subjects
    
    if use_tfce:
        # FWER Correction: Return max statistic
        null_mean_3d = np.zeros(mask.shape, dtype=np.float32)
        null_mean_3d[mask] = null_mean
        null_mean_3d = apply_tfce(null_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
        return np.max(np.abs(null_mean_3d))
    else:
        return null_mean


def parse_args():
    parser = argparse.ArgumentParser(description='Step 2: Statistical Analysis for ISFC')
    parser.add_argument('--input_map', type=str, 
                        help='Path to 4D ISFC map (Z-score recommended for T-test/Bootstrap). Required for T-test/Bootstrap.')
    parser.add_argument('--method', type=str, choices=['ttest', 'bootstrap', 'phaseshift'], required=True,
                        help='Statistical method: "ttest", "bootstrap", or "phaseshift"')
    parser.add_argument('--condition', type=str, 
                        help='Condition name. Required for Phase Shift.')
    parser.add_argument('--roi_id', type=int, default=None,
                        help='ROI ID (if using Phase Shift or masking input map)')
    parser.add_argument('--n_perms', type=int, default=1000,
                        help='Number of permutations/bootstraps (default: 1000)')
    parser.add_argument('--p_threshold', type=float, default=0.05,
                        help='P-value threshold (default: 0.05)')
    parser.add_argument('--cluster_threshold', type=int, default=0,
                        help='Cluster extent threshold (min voxels). Default: 0')
    parser.add_argument('--use_tfce', action='store_true',
                        help='Use Threshold-Free Cluster Enhancement (requires permutation/bootstrap). Incompatible with cluster_threshold.')
    parser.add_argument('--tfce_E', type=float, default=0.5,
                        help='TFCE extent parameter (default: 0.5)')
    parser.add_argument('--tfce_H', type=float, default=2.0,
                        help='TFCE height parameter (default: 2.0)')
    parser.add_argument('--seed_x', type=float, help='Seed X (Required for Phase Shift)')
    parser.add_argument('--seed_y', type=float, help='Seed Y (Required for Phase Shift)')
    parser.add_argument('--seed_z', type=float, help='Seed Z (Required for Phase Shift)')
    parser.add_argument('--seed_radius', type=float, default=5, help='Seed Radius (Required for Phase Shift)')
    parser.add_argument('--seed_file', type=str, help='Seed File (Optional for Phase Shift)')
    
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
    Run one-sample T-test against 0 on the last dimension of data.
    """
    print("Running T-test...")
    t_stats, p_values = ttest_1samp(data_4d, popmean=0, axis=-1, nan_policy='omit')
    mean_map = np.nanmean(data_4d, axis=-1)
    
    return mean_map, p_values

def run_bootstrap(data_4d, n_bootstraps=1000, random_state=42, use_tfce=False, mask_3d=None, tfce_E=0.5, tfce_H=2.0):
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

    observed_mean = np.nanmean(data_4d, axis=1) # (V,)
    
    if use_tfce:
        if mask_3d is None:
            raise ValueError("mask_3d is required when use_tfce=True")
        # Reshape observed mean to 3D for TFCE
        observed_mean_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        observed_mean_3d[mask_3d] = observed_mean
        observed_mean_3d = apply_tfce(observed_mean_3d, mask_3d, E=tfce_E, H=tfce_H, two_sided=True)
        observed_mean = observed_mean_3d[mask_3d]
    
    data_centered = data_4d - np.nanmean(data_4d, axis=1, keepdims=True)
    
    # Parallelize
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_bootstrap_iter)(
            i, n_samples, data_centered, use_tfce, mask_3d, tfce_E, tfce_H, random_state + i
        ) for i in range(n_bootstraps)
    )
    
    if use_tfce:
        # FWER Correction
        null_max_stats = np.array(results) 
        p_values = (np.sum(null_max_stats[np.newaxis, :] >= np.abs(observed_mean[:, np.newaxis]), axis=1) + 1) / (n_bootstraps + 1)
    else:
        # Voxel-wise
        null_means = np.array(results).T
        with np.errstate(invalid='ignore'):
             p_values = np.sum(np.abs(null_means) >= np.abs(observed_mean[:, np.newaxis]), axis=1) / (n_bootstraps + 1)
    
    return observed_mean, p_values


def run_phaseshift(condition, roi_id, seed_coords, seed_radius, n_perms, data_dir, mask_file, chunk_size=config.CHUNK_SIZE, seed_file=None, use_tfce=False, tfce_E=0.5, tfce_H=2.0):
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
    
    if seed_file:
         seed_mask_data, _ = load_mask(seed_file)
         if seed_mask_data.shape != mask.shape:
             raise ValueError("Seed file shape mismatch")
         seed_mask = seed_mask_data > 0
    else:
         seed_mask = get_seed_mask(mask.shape, affine, seed_coords, seed_radius)
         
    obs_seed_ts = load_seed_data(group_data, seed_mask, mask)
    
    print("  Generating surrogate seeds...")
    
    # 1. Observed
    print("  Computing Observed ISFC...")
    obs_isfc_raw, obs_isfc_z = run_isfc_computation(group_data, obs_seed_ts, pairwise=False, chunk_size=chunk_size)
    obs_mean_z = np.nanmean(obs_isfc_z, axis=1) # (V,)
    
    if use_tfce:
        # Apply TFCE to observed map
        obs_mean_z_3d = np.zeros(mask.shape, dtype=np.float32)
        obs_mean_z_3d[mask] = obs_mean_z
        obs_mean_z_3d = apply_tfce(obs_mean_z_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
        obs_mean_z = obs_mean_z_3d[mask]
    
    # 2. Null Distribution (Parallel)
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_phaseshift_iter)(
            i, obs_seed_ts, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, 1000 + i
        ) for i in range(n_perms)
    )
    
    if use_tfce:
        # FWER Correction
        null_max_stats = np.array(results)
        p_values = (np.sum(null_max_stats[np.newaxis, :] >= np.abs(obs_mean_z[:, np.newaxis]), axis=1) + 1) / (n_perms + 1)
    else:
         # Voxel-wise
         null_means = np.array(results).T
         count = np.sum(np.abs(null_means) >= np.abs(obs_mean_z[:, np.newaxis]), axis=1)
         p_values = (count + 1) / (n_perms + 1)
    
    # Convert back to 3D for return
    obs_mean_z_3d = np.zeros(mask.shape, dtype=np.float32)
    obs_mean_z_3d[mask] = obs_mean_z
    
    p_values_3d = np.ones(mask.shape, dtype=np.float32)
    p_values_3d[mask] = p_values
    
    return obs_mean_z_3d, p_values_3d, mask, affine

def main():
    args = parse_args()
    method = args.method
    roi_id = args.roi_id
    threshold = args.p_threshold
    output_dir = args.output_dir
    data_dir = args.data_dir
    mask_file = args.mask_file
    chunk_size = args.chunk_size
    
    print(f"--- Step 2: ISFC Statistics ---")
    print(f"Method: {method}")
    print(f"Threshold: {threshold}")
    print(f"Output Dir: {output_dir}")
    print(f"Data Dir: {data_dir}")
    print(f"Chunk Size: {chunk_size}")

    # ... (initialization) ...
    
    mask_affine = None
    mask_data = None
    mean_map = None
    p_values = None
    
    if args.roi_id is not None: 
         mask_data, mask_affine = load_mask(mask_file, roi_id=args.roi_id)

    if method == 'phaseshift':
        if not args.condition:
            raise ValueError("Phaseshift requires --condition")
            
        seed_coords = None
        seed_radius = args.seed_radius
        
        if args.seed_file:
             print(f"Using seed file: {args.seed_file}")
             seed_suffix = f"_{os.path.basename(args.seed_file).replace('.nii', '').replace('.gz', '')}"
        elif args.seed_x is not None:
             seed_coords = (args.seed_x, args.seed_y, args.seed_z)
             print(f"Using seed coordinates: {seed_coords} (r={seed_radius}mm)")
             seed_suffix = f"_seed{int(seed_coords[0])}_{int(seed_coords[1])}_{int(seed_coords[2])}_r{int(seed_radius)}"
        else:
             raise ValueError("Phaseshift requires --seed_file OR --seed_x/y/z")
             
        mean_map, p_values, mask_data, mask_affine = run_phaseshift(
            args.condition, args.roi_id, seed_coords, args.seed_radius, args.n_perms, 
            data_dir=data_dir, mask_file=mask_file, chunk_size=chunk_size, seed_file=args.seed_file,
            use_tfce=args.use_tfce, tfce_E=args.tfce_E, tfce_H=args.tfce_H
        )
        base_name = f"isfc_{args.condition}_{method}{seed_suffix}"

        
    else: # Map based
        if not args.input_map:
            raise ValueError("Map-based stats require --input_map")
            
        print(f"Loading input map: {args.input_map}")
        img = nib.load(args.input_map)
        data_4d = img.get_fdata(dtype=np.float32)
        if mask_affine is None: mask_affine = img.affine
        
        # If mask_data is None (no ROI specified), we create one from non-zero?
        # Or we use the loaded mask.
        if mask_data is None:
             # Load whole brain mask
             mask_data, _ = load_mask(mask_file, roi_id=None)
             
        # Apply mask to data_4d if needed (flattening) or just process 4D?
        # T-test/Bootstrap works on arrays.
        # If data_4d is (X,Y,Z,S), and mask is (X,Y,Z).
        # We want to process only valid voxels to save time/memory.
        
        masked_data = data_4d[mask_data] # (V, S)
        
        if method == 'ttest':
            if args.use_tfce:
                print("Warning: TFCE requires permutation/bootstrap. T-test does not support TFCE. Ignoring --use_tfce.")
            mean_vals, p_vals_vec = run_ttest(masked_data)
        elif method == 'bootstrap':
            mean_vals, p_vals_vec = run_bootstrap(
                masked_data, n_bootstraps=args.n_perms,
                use_tfce=args.use_tfce, mask_3d=mask_data,
                tfce_E=args.tfce_E, tfce_H=args.tfce_H
            )
            
        # Reconstruct maps
        mean_map = np.zeros(mask_data.shape, dtype=np.float32)
        mean_map[mask_data] = mean_vals
        
        p_values = np.ones(mask_data.shape, dtype=np.float32)
        p_values[mask_data] = p_vals_vec
        
        input_base = os.path.basename(args.input_map).replace('.nii.gz', '').replace('_desc-zscore', '').replace('_desc-raw', '')
        base_name = f"{input_base}_{method}"
    
    # TFCE suffix
    tfce_suffix = "_tfce" if args.use_tfce else ""
    if tfce_suffix:
        base_name += tfce_suffix
    
    # Check incompatibility
    if args.use_tfce and args.cluster_threshold > 0:
        print("Warning: TFCE and cluster_threshold are incompatible. Ignoring cluster_threshold.")
        args.cluster_threshold = 0

    # Save Outputs
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 1a. Un-thresholded Map (TFCE score or Mean Statistic)
    stat_suffix = "tfce" if args.use_tfce else "stat"
    stat_path = os.path.join(output_dir, f"{base_name}_desc-{stat_suffix}.nii.gz")
    save_map(mean_map, mask_data, mask_affine, stat_path)

    # 1b. P-values
    p_path = os.path.join(output_dir, f"{base_name}_desc-pvalues.nii.gz")
    save_map(p_values, mask_data, mask_affine, p_path)
    
    # 2. Significant Map
    sig_map = mean_map.copy()
    sig_map[p_values >= threshold] = 0
    
    if args.cluster_threshold > 0:
        sig_map = apply_cluster_threshold(sig_map, args.cluster_threshold)
        
    clust_suffix = f"_clust{args.cluster_threshold}" if args.cluster_threshold > 0 else ""
    sig_path = os.path.join(output_dir, f"{base_name}_desc-sig_p{str(threshold).replace('.', '')}{clust_suffix}.nii.gz")
    save_map(sig_map, mask_data, mask_affine, sig_path)
    
    # 3. Plot
    plot_path = os.path.join(output_dir, f"{base_name}_desc-sig.png")
    save_plot(sig_path, plot_path, f"Sig ISFC ({method}, p<{threshold})", positive_only=True)
    
    print("Done")
    print(f"Outputs:\n  {p_path}\n  {sig_path}\n  {plot_path}")

if __name__ == "__main__":
    main()
