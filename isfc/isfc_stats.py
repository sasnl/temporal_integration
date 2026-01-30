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
# Removed brainiak.isc.bootstrap_isc import as we are implementing it manually for parallelization

def _run_bootstrap_iter(i, n_samples, data_centered, use_tfce, mask_3d, tfce_E, tfce_H, tfce_dh, seed):
    """
    Helper function for parallel bootstrap iteration (Median).
    """
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n_samples, size=n_samples)
    sample = data_centered[:, indices]
    
    # Compute Median for this bootstrap sample
    perm_stat = np.nanmedian(sample, axis=1)
    
    if use_tfce:
        # Apply TFCE to permuted map (relative to null)
        perm_stat_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        perm_stat_3d[mask_3d] = perm_stat
        
        # TFCE on null distribution (two-sided usually captures magnitude)
        perm_stat_3d = apply_tfce(perm_stat_3d, mask_3d, E=tfce_E, H=tfce_H, dh=tfce_dh, two_sided=False)
        
        # Return Max-Statistic for FWER correction
        return np.max(np.abs(perm_stat_3d))
    else:
        # For non-TFCE FWER (Max-Stat), we need the max statistic of the map
        # But for voxel-wise p-values, we need the whole map.
        # To support both efficiently, let's return the map (V,)
        return perm_stat



def _run_phaseshift_iter(i, n_perms, obs_seed_ts, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, tfce_dh, seed):
    if (i+1) % 10 == 0 or i == 0:
        print(f"Starting permutation {i+1} out of {n_perms}", flush=True)
    # Generate surrogate seed by phase randomizing the OBSERVED seed
    surr_seed_ts = phase_randomize(obs_seed_ts, voxelwise=False, random_state=seed)
    
    surr_raw, surr_z = run_isfc_computation(group_data, surr_seed_ts, pairwise=False, chunk_size=chunk_size)
    null_mean = np.nanmean(surr_z, axis=1) # Mean over subjects
    
    if use_tfce:
        # FWER Correction: Return max statistic
        null_mean_3d = np.zeros(mask.shape, dtype=np.float32)
        null_mean_3d[mask] = null_mean
        null_mean_3d = apply_tfce(null_mean_3d, mask, E=tfce_E, H=tfce_H, dh=tfce_dh, two_sided=False)
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
    parser.add_argument('--tfce_dh', type=float, default=0.01,
                        help='TFCE step size. Default: 0.01 (finer step for Z-scores)')
    parser.add_argument('--fwe_method', type=str, choices=['none', 'max_stat', 'bonferroni', 'fdr'], default='none',
                        help='Family-Wise Error correction method for non-TFCE stats. Choices: "none", "max_stat", "bonferroni", "fdr". Default: "none".')
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
    parser.add_argument('--save_permutations', action='store_true',
                        help='Save all permutation maps to disk')
    return parser.parse_args()

def run_ttest(data_4d):
    """
    Run one-sample T-test against 0 on the last dimension of data.
    """
    print("Running T-test...")
    t_stats, p_values = ttest_1samp(data_4d, popmean=0, axis=-1, nan_policy='omit')
    mean_map = np.nanmean(data_4d, axis=-1)
    
    return mean_map, p_values

def run_bootstrap(data_4d, n_bootstraps=1000, random_state=42, use_tfce=False, mask_3d=None, tfce_E=0.5, tfce_H=2.0, tfce_dh=0.01, fwe_method='none', save_permutations=False):
    """
    Run bootstrap manually with parallelization (Joblib).
    Replaces run_bootstrap_brainiak for performance reasons.
    
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
    fwe_method : str
        'none', 'max_stat', or 'bonferroni'. Used if use_tfce is False.
    """
    print(f"Running Parallel Bootstrap (n={n_bootstraps}, summary=median, side=right)...")
    if use_tfce:
        print("Note: TFCE implies FWER correction via Max-TFCE. Ignoring --fwe_method arguments.")
    elif fwe_method != 'none':
        print(f"Applying FWER correction method: {fwe_method}")
        
    n_voxels, n_samples = data_4d.shape
    
    # 1. Compute Observed Statistic (Median)
    # -------------------------------------
    observed_check = np.nanmedian(data_4d, axis=1) # (V,)
    
    # 2. Shift Data to Null Hypothesis
    # --------------------------------
    # Hall & Wilson (1991) Geometric shift: Center data so pop median is 0
    # Center each voxel's time series (samples) by subtracting its median
    data_centered = data_4d - np.nanmedian(data_4d, axis=1, keepdims=True)
    
    # 3. Parallel Bootstrap Resampling
    # --------------------------------
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_bootstrap_iter)(
            i, n_samples, data_centered, use_tfce, mask_3d, tfce_E, tfce_H, tfce_dh, random_state + i
        ) for i in range(n_bootstraps)
    )
    
    # 4. Compute P-values
    # -------------------
    
    if use_tfce:
        # TFCE FWER Correction
        # results contains list of Max-TFCE stats (one per boot)
        
        # 3a. Compute Observed TFCE
        if mask_3d is None: raise ValueError("mask_3d required for TFCE")
        obs_map_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        obs_map_3d[mask_3d] = observed_check
        obs_tfce_3d = apply_tfce(obs_map_3d, mask_3d, E=tfce_E, H=tfce_H, dh=tfce_dh, two_sided=False)
        obs_tfce_flat = obs_tfce_3d[mask_3d]
        
        # 3b. Null Max Stats
        null_max_stats = np.array(results) # (n_boots,)
        
        # 3c. P-values (Right Tailed)
        # p = (sum(Max_Null >= Obs) + 1) / (B + 1)
        sorted_max_stats = np.sort(null_max_stats)
        indices = np.searchsorted(sorted_max_stats, obs_tfce_flat, side='left')
        count_greater = n_bootstraps - indices
        p_values_corrected = (count_greater + 1) / (n_bootstraps + 1)
        
        # Uncorrected not easily available from max stats, returning ones or recompute?
        # Recomputing uncorrected p-values would require returning whole maps which is heavy.
        # We will skip valid uncorrected P-values for TFCE optimization or just return corrected.
        p_values_uncorrected = p_values_corrected # Placeholder or logic change needed? 
        # Actually for TFCE, users mostly care about corrected.
        
        return obs_tfce_flat, p_values_corrected, p_values_uncorrected, None

    else:
        # Non-TFCE
        null_dist_maps = np.array(results).T # (V, n_boots)
        
        # Uncorrected P-values (Voxel-wise)
        # Count how many null values >= observed value (Right-Tailed)
        count_greater = np.sum(null_dist_maps >= observed_check[:, np.newaxis], axis=1)
        p_uncorrected = (count_greater + 1) / (n_bootstraps + 1)
        
        if fwe_method == 'max_stat':
            print("Computing Max-Statistic FWER correction...")
            # Max statistic across voxels for each bootstrap iteration
            max_stats = np.max(null_dist_maps, axis=0) # (n_boots,)
            
            sorted_max_stats = np.sort(max_stats)
            indices = np.searchsorted(sorted_max_stats, observed_check, side='left')
            count_greater_corr = n_bootstraps - indices
            p_corrected = (count_greater_corr + 1) / (n_bootstraps + 1)
            
            return observed_check, p_corrected, p_uncorrected, null_dist_maps if save_permutations else None
            

            
        elif fwe_method == 'fdr':
            print("Applying FDR correction (Benjamini-Hochberg)...")
            # Flatten for sorting
            p_flat = p_uncorrected.flatten()
            n_vals = p_flat.size
            sorted_indices = np.argsort(p_flat)
            sorted_p = p_flat[sorted_indices]
            
            ranks = np.arange(1, n_vals + 1)
            q_vals = sorted_p * n_vals / ranks
            
            # Monotonicity
            q_vals[-1] = min(q_vals[-1], 1.0)
            for i in range(n_vals - 2, -1, -1):
                q_vals[i] = min(q_vals[i], q_vals[i+1])
                
            p_corrected_flat = np.zeros_like(p_flat)
            p_corrected_flat[sorted_indices] = q_vals
            
            return observed_check, p_corrected_flat, p_uncorrected, null_dist_maps if save_permutations else None

        elif fwe_method == 'bonferroni':
            print("Applying Bonferroni correction...")
            n_voxels = observed_check.shape[0]
            p_corrected = p_uncorrected * n_voxels
            p_corrected[p_corrected > 1] = 1.0
            return observed_check, p_corrected, p_uncorrected, null_dist_maps if save_permutations else None
            
        else:
            return observed_check, p_uncorrected, p_uncorrected, null_dist_maps if save_permutations else None


def run_phaseshift(condition, roi_id, seed_coords, seed_radius, n_perms, data_dir, mask_file, chunk_size=config.CHUNK_SIZE, seed_file=None, use_tfce=False, tfce_E=0.5, tfce_H=2.0, tfce_dh=0.01, save_permutations=False):
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
        obs_mean_z_3d = apply_tfce(obs_mean_z_3d, mask, E=tfce_E, H=tfce_H, dh=tfce_dh, two_sided=False)
        obs_mean_z = obs_mean_z_3d[mask]
    
    # 2. Null Distribution (Parallel)
    print(f"  Starting {n_perms} permutations...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_phaseshift_iter)(
            i, n_perms, obs_seed_ts, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, tfce_dh, 1000 + i
        ) for i in range(n_perms)
    )
    
    if use_tfce:
        # FWER Correction
        null_max_stats = np.array(results)
        p_values = (np.sum(null_max_stats[np.newaxis, :] >= np.abs(obs_mean_z[:, np.newaxis]), axis=1) + 1) / (n_perms + 1)
    else:
         # Voxel-wise
         null_means = np.array(results).T # (V, n_perms)
         count = np.sum(np.abs(null_means) >= np.abs(obs_mean_z[:, np.newaxis]), axis=1)
         p_values = (count + 1) / (n_perms + 1)
    
    # Check save_permutations
    if save_permutations and not use_tfce:
        # For ISFC phase shift, we usually have null_means available in 'results' (list of V arrays) or 'null_means'
        # If use_tfce was True, results are max stats, not maps. So we can't save permutations for offline TFCE if we ran with TFCE.
        # But if use_tfce is False (voxel-wise), we have the maps.
        if 'null_means' in locals():
            perm_maps = null_means
        else:
            # Reconstruct from results if needed (for tfce case it implies we didn't save maps, just max)
            # Actually if use_tfce=True, 'results' contains floats, so we CANNOT save perm maps unless we modify _run_phaseshift_iter to return both.
            # For now, only support saving if use_tfce=False.
            perm_maps = np.array(results).T
    else:
        perm_maps = None
    
    # Convert back to 3D for return
    obs_mean_z_3d = np.zeros(mask.shape, dtype=np.float32)
    obs_mean_z_3d[mask] = obs_mean_z
    
    p_values_3d = np.ones(mask.shape, dtype=np.float32)
    p_values_3d[mask] = p_values
    
    return obs_mean_z_3d, p_values_3d, mask, affine, perm_maps

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
    p_uncorrected = None
    p_uncorrected_3d = None
    
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
             
        res = run_phaseshift(
            args.condition, args.roi_id, seed_coords, args.seed_radius, args.n_perms, 
            data_dir=data_dir, mask_file=mask_file, chunk_size=chunk_size, seed_file=args.seed_file,
            use_tfce=args.use_tfce, tfce_E=args.tfce_E, tfce_H=args.tfce_H, tfce_dh=args.tfce_dh,
            save_permutations=args.save_permutations
        )
        # Unpack depending on what run_phaseshift returns. 
        if len(res) == 5:
            mean_map, p_values, mask_data, mask_affine, perm_maps = res
        else:
             mean_map, p_values, mask_data, mask_affine = res
             perm_maps = None
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
        
        perm_maps = None
        
        if method == 'ttest':
            if args.use_tfce:
                print("Warning: TFCE requires permutation/bootstrap. T-test does not support TFCE. Ignoring --use_tfce.")
            mean_vals, p_vals_vec = run_ttest(masked_data)
        elif method == 'bootstrap':
            mean_vals, p_vals_vec, p_uncorrected, perm_maps_flat = run_bootstrap(
                masked_data, n_bootstraps=args.n_perms,
                use_tfce=args.use_tfce, mask_3d=mask_data,
                tfce_E=args.tfce_E, tfce_H=args.tfce_H, tfce_dh=args.tfce_dh,
                fwe_method=args.fwe_method, save_permutations=args.save_permutations
            )
            
            # Reconstruct perm maps
            if perm_maps_flat is not None:
                perm_maps = perm_maps_flat # (V, N)
            else:
                perm_maps = None
            
        # Reconstruct maps
        mean_map = np.zeros(mask_data.shape, dtype=np.float32)
        mean_map[mask_data] = mean_vals
        
        p_values = np.ones(mask_data.shape, dtype=np.float32)
        p_values[mask_data] = p_vals_vec
        
        if method == 'bootstrap':
            p_uncorrected_3d = np.ones(mask_data.shape, dtype=np.float32)
            p_uncorrected_3d[mask_data] = p_uncorrected
        
        input_base = os.path.basename(args.input_map).replace('.nii.gz', '').replace('_desc-zscore', '').replace('_desc-raw', '')
        base_name = f"{input_base}_{method}"
    
    # TFCE suffix
    tfce_suffix = "_tfce" if args.use_tfce else ""
    if tfce_suffix:
        base_name += tfce_suffix
        
    # FWER suffix
    if args.fwe_method != 'none' and not args.use_tfce:
        base_name += f"_{args.fwe_method}"
    
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
    
    # 1c. Uncorrected P-values
    if p_uncorrected_3d is not None:
        p_unc_path = os.path.join(output_dir, f"{base_name}_desc-pvalues_uncorrected.nii.gz")
        save_map(p_uncorrected_3d, mask_data, mask_affine, p_unc_path)
    
    # 1d. Permutations (if requested and available)
    if perm_maps is not None:
        print(f"Saving all {args.n_perms} permutation maps to disk...")
        n_perms = perm_maps.shape[1]
        perm_maps_4d = np.zeros(mask_data.shape + (n_perms,), dtype=np.float32)
        perm_maps_4d[mask_data, :] = perm_maps
        
        perm_path = os.path.join(output_dir, f"{base_name}_desc-perms.nii.gz")
        nib.save(nib.Nifti1Image(perm_maps_4d, mask_affine), perm_path)
        print(f"Saved permutations to: {perm_path}")
    
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
