import os
import sys
import argparse
import numpy as np
import time
import nibabel as nib
from joblib import Parallel, delayed
from scipy.stats import ttest_1samp
from brainiak.utils.utils import phase_randomize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TI_CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if TI_CODE_DIR not in sys.path:
    sys.path.insert(0, TI_CODE_DIR)

from shared import config
from shared.pipeline_utils import (
    load_mask, load_data, save_map, save_plot, run_isc_computation,
    apply_cluster_threshold, apply_tfce
)

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
    parser.add_argument("--checkpoint_every", type=int, default=25,
                        help="Save phaseshift checkpoint every N permutations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume phaseshift from existing checkpoint in output_dir")

    
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

def _run_phaseshift_iter(i, n_perms, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, seed):
    if (i+1) % 10 == 0 or i == 0:
        print(f"Starting permutation {i+1} out of {n_perms}", flush=True)
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
        null_mean_3d = apply_tfce(null_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=False)
        return np.max(np.abs(null_mean_3d))
    else:
        return null_mean

# def _run_phaseshift_iter(i, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, seed):
#     n_subs = group_data.shape[2]
#     rng = np.random.RandomState(seed)
    
#     shifted_data = np.zeros_like(group_data)
#     for s in range(n_subs):
#         shifted_data[:, :, s] = phase_randomize(group_data[:, :, s], random_state=rng)
        
#     null_raw, null_z = run_isc_computation(shifted_data, chunk_size=chunk_size)
#     null_mean = np.nanmean(null_z, axis=1)
    
#     if use_tfce:
#         null_mean_3d = np.zeros(mask.shape, dtype=np.float32)
#         null_mean_3d[mask] = null_mean
#         null_mean_3d = apply_tfce(null_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
#         return np.max(np.abs(null_mean_3d))
#     else:
#         return null_mean

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
    
def _run_bootstrap_iter(i, n_samples, data_centered, use_tfce, mask_3d, tfce_E, tfce_H, seed):
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, n_samples, size=n_samples)
    sample = data_centered[:, indices]
    perm_mean = np.nanmean(sample, axis=1)
    
    if use_tfce:
        perm_mean_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        perm_mean_3d[mask_3d] = perm_mean
        perm_mean_3d = apply_tfce(perm_mean_3d, mask_3d, E=tfce_E, H=tfce_H, two_sided=True)
        return np.max(np.abs(perm_mean_3d))
    else:
        return perm_mean


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
    
    # Center data
    data_centered = data_4d - np.nanmean(data_4d, axis=1, keepdims=True)
    
    # Parallelize
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_bootstrap_iter)(
            i, n_samples, data_centered, use_tfce, mask_3d, tfce_E, tfce_H, random_state + i
        ) for i in range(n_bootstraps)
    )
    
    if use_tfce:
        # Results are list of max-stats (one per bootstrap)
        null_max_stats = np.array(results) # (n_bootstraps,)
    else:
        # Results are list of arrays (n_voxels,)
        null_means = np.array(results).T # (n_voxels, n_bootstraps)
        
    # P-value: Proportion of null means >= observed mean (two-sided: use absolute values)
    with np.errstate(invalid='ignore'):
         p_values = np.sum(np.abs(null_means) >= np.abs(observed_mean[:, np.newaxis]), axis=1) / (n_bootstraps + 1)
    
    return observed_mean, p_values

    
#DE added check points to be able to pick up where we left off
    
def run_phaseshift(condition, roi_id, n_perms, data_dir, mask_file, output_dir, chunk_size=config.CHUNK_SIZE, use_tfce=False, tfce_E=0.5, tfce_H=2.0, checkpoint_every=25,resume=False):
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
    obs_mean = np.nanmean(obs_z, axis=1).astype(np.float32)  # (V,)
    
    if use_tfce:
        # Apply TFCE to observed map
        obs_mean_3d = np.zeros(mask.shape, dtype=np.float32)
        obs_mean_3d[mask] = obs_mean
        obs_mean_3d = apply_tfce(obs_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=False)
        obs_mean = obs_mean_3d[mask].astype(np.float32)
    
    # 2. Phase Randomization (Parallel)
    # print(f"  Starting {n_perms} permutations...")
    # results = Parallel(n_jobs=-1, verbose=5)(
    #     delayed(_run_phaseshift_iter)(
    #         i, n_perms, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, 1000 + i
    #     ) for i in range(n_perms)
    # )


    if output_dir is None:
        raise ValueError("output_dir is required for checkpointing and resume")

    nullmaps_dir = os.path.join(output_dir, "null_maps")
    os.makedirs(nullmaps_dir, exist_ok=True)

    ckpt_name = f"isc_{condition}_phaseshift"
    if roi_id is not None:
        ckpt_name += f"_roi-{roi_id}"
    if use_tfce:
        ckpt_name += "_tfce"
    ckpt_path = os.path.join(output_dir, f"{ckpt_name}_ckpt.npz")

    completed = 0

    if use_tfce:
        null_max_stats = []
    else:
        count_greater = np.zeros(n_voxels, dtype=np.int32)
        abs_obs = np.abs(obs_mean).astype(np.float32)

    if resume and os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        completed = int(ck["completed"])

        meta = ck["meta"].item()
        expected = {
            "condition": condition,
            "roi_id": roi_id,
            "chunk_size": int(chunk_size),
            "use_tfce": bool(use_tfce),
            "tfce_E": float(tfce_E),
            "tfce_H": float(tfce_H),
            "n_voxels": int(n_voxels),
        }
        if meta != expected:
            raise ValueError(f"Checkpoint meta mismatch.\nFound: {meta}\nExpected: {expected}")

        if use_tfce:
            null_max_stats = ck["null_max_stats"].astype(np.float32).tolist()
        else:
            count_greater = ck["count_greater"].astype(np.int32)

        print(f"Resuming from checkpoint {ckpt_path}")
        print(f"Completed permutations: {completed} / {n_perms}")

    print(f"  Starting {n_perms} permutations...")

    for b0 in range(completed, n_perms, checkpoint_every):
        b1 = min(b0 + checkpoint_every, n_perms)
        print(f"  Running permutations {b0} to {b1 - 1}")

        batch_results = Parallel(n_jobs=-1, verbose=5)(
            delayed(_run_phaseshift_iter)(
                i, n_perms, group_data, chunk_size, use_tfce, mask, tfce_E, tfce_H, 1000 + i
            ) for i in range(b0, b1)
        )

        if use_tfce:
            null_max_stats.extend([float(x) for x in batch_results])
        else:
            batch_null_means = []
            for null_mean in batch_results:
                null_mean = null_mean.astype(np.float32)
                batch_null_means.append(null_mean)
                count_greater += (np.abs(null_mean) >= abs_obs).astype(np.int32)

            batch_null_means = np.stack(batch_null_means, axis=1)  # (n_voxels, batch_size)

            batch_fname = f"nullmeans_perm{b0:04d}_{b1-1:04d}.npz"
            batch_path = os.path.join(nullmaps_dir, batch_fname)

            if not os.path.exists(batch_path):
                np.savez_compressed(
                batch_path,
                null_means=batch_null_means,
                perm_start=b0,
                perm_end=b1 - 1
            )
                print(f"  Saved null maps: {batch_path}")
            else:
                print(f"  Null maps already exist, skipping save: {batch_path}")

        completed = b1

        meta = {
            "condition": condition,
            "roi_id": roi_id,
            "chunk_size": int(chunk_size),
            "use_tfce": bool(use_tfce),
            "tfce_E": float(tfce_E),
            "tfce_H": float(tfce_H),
            "n_voxels": int(n_voxels),
        }

        if use_tfce:
            np.savez(
                ckpt_path,
                completed=completed,
                null_max_stats=np.array(null_max_stats, dtype=np.float32),
                meta=meta
            )
        else:
            np.savez(
                ckpt_path,
                completed=completed,
                count_greater=count_greater,
                meta=meta
            )

        print(f"  Saved checkpoint at {completed}: {ckpt_path}")


    # if use_tfce:
    #     # FWER Correction
    #     null_max_stats = np.array(results) # (n_perms,)
    #     p_values = (np.sum(null_max_stats[np.newaxis, :] >= np.abs(obs_mean[:, np.newaxis]), axis=1) + 1) / (n_perms + 1)
    # else:
    #     # Voxel-wise
    #     null_means = np.array(results).T # (V, n_perms)
    #     p_values = np.sum(np.abs(null_means) >= np.abs(obs_mean[:, np.newaxis]), axis=1) / (n_perms + 1)
    if use_tfce:
        null_max_arr = np.array(null_max_stats, dtype=np.float32)
        p_values = (
            (np.sum(null_max_arr[np.newaxis, :] >= np.abs(obs_mean[:, np.newaxis]), axis=1) + 1)
            / (len(null_max_arr) + 1)
        ).astype(np.float32)
    else:
        p_values = ((count_greater + 1) / (completed + 1)).astype(np.float32)
    
    
    # Convert back to 3D for return
    obs_mean_3d = np.zeros(mask.shape, dtype=np.float32)
    obs_mean_3d[mask] = obs_mean
    
    p_values_3d = np.ones(mask.shape, dtype=np.float32)
    p_values_3d[mask] = p_values
    
    return obs_mean_3d, p_values_3d, mask, affine 
# def run_phaseshift(condition, roi_id, n_perms, data_dir, mask_file, chunk_size=config.CHUNK_SIZE, use_tfce=False, tfce_E=0.5, tfce_H=2.0):
#     """
#     Run Phase Shift randomization.
    
#     Parameters:
#     -----------
#     use_tfce : bool
#         If True, apply TFCE transformation before computing p-values
#     tfce_E : float
#         TFCE extent parameter
#     tfce_H : float
#         TFCE height parameter
#     """
#     print(f"Running Phase Shift (n={n_perms}, chunk_size={chunk_size})...")
    
#     mask, affine = load_mask(mask_file, roi_id=roi_id)
#     if np.sum(mask) == 0: raise ValueError("Empty mask")
    
#     group_data = load_data(condition, config.SUBJECTS, mask, data_dir)
#     if group_data is None: raise ValueError("No data")
    
#     n_trs, n_voxels, n_subs = group_data.shape
    
#     # 1. Compute Observed ISC
#     obs_raw, obs_z = run_isc_computation(group_data, chunk_size=chunk_size)
#     obs_mean = np.nanmean(obs_z, axis=1) # (V,)
    
#     if use_tfce:
#         # Apply TFCE to observed map
#         obs_mean_3d = np.zeros(mask.shape, dtype=np.float32)
#         obs_mean_3d[mask] = obs_mean
#         obs_mean_3d = apply_tfce(obs_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
#         obs_mean = obs_mean_3d[mask]
    
#     # 2. Phase Randomization
#     count_greater = np.zeros(n_voxels, dtype=int)
    
#     rng = np.random.RandomState(42)
    
#     for i in range(n_perms):
#         if i % 10 == 0: print(f"  Permutation {i}/{n_perms}")
        
#         # Shift each subject
#         shifted_data = np.zeros_like(group_data)
#         for s in range(n_subs):
#             shifted_data[:, :, s] = phase_randomize(group_data[:, :, s], random_state=rng)
            
#         # Compute Null ISC
#         null_raw, null_z = run_isc_computation(shifted_data, chunk_size=chunk_size)
#         null_mean = np.nanmean(null_z, axis=1)
        
#         if use_tfce:
#             # Apply TFCE to permuted map
#             null_mean_3d = np.zeros(mask.shape, dtype=np.float32)
#             null_mean_3d[mask] = null_mean
#             null_mean_3d = apply_tfce(null_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
#             null_mean = null_mean_3d[mask]
        
#         count_greater += (np.abs(null_mean) >= np.abs(obs_mean))
        
#     p_values = (count_greater + 1) / (n_perms + 1)
    
#     # Convert back to 3D for return
#     obs_mean_3d = np.zeros(mask.shape, dtype=np.float32)
#     obs_mean_3d[mask] = obs_mean
    
#     p_values_3d = np.ones(mask.shape, dtype=np.float32)
#     p_values_3d[mask] = p_values
    
#     return obs_mean_3d, p_values_3d, mask, affine 

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
        mean_map, p_values, mask_data, mask_affine = run_phaseshift(
            args.condition, roi_id, args.n_perms, 
            data_dir=data_dir, mask_file=mask_file,   output_dir=output_dir, chunk_size=chunk_size,
            use_tfce=args.use_tfce, tfce_E=args.tfce_E, tfce_H=args.tfce_H,checkpoint_every=args.checkpoint_every,resume=args.resume,  
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
