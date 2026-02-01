import os
import sys
import argparse
import numpy as np
import nibabel as nib
from scipy.stats import ttest_1samp
from brainiak.utils.utils import phase_randomize
from joblib import Parallel, delayed
from dask.distributed import progress


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TI_CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if TI_CODE_DIR not in sys.path:
    sys.path.insert(0, TI_CODE_DIR)

from isfc import config
print(f"CONFIG MODULE FILE: {os.path.abspath(config.__file__)}", flush=True)

from isfc.pipeline_utils_dist import (
    load_mask,
    load_data,
    save_map,
    save_plot,
    get_seed_mask,
    load_seed_data,
    apply_cluster_threshold,
    apply_tfce,run_isfc_computation
)


from dask_jobqueue import SLURMCluster
from distributed import Client,LocalCluster
from dask import delayed as dask_delayed

def _compute_loo_null_isfc(group_data, obs_seed_ts, rng):
    """
    Compute LOO null ISFC using the correct approach:
    - For each subject: phase randomize ONLY that subject's seed
    - Compare the shifted seed to the mean target data of OTHER (unshifted) subjects
    - Roll through all subjects

    Parameters:
    -----------
    group_data : ndarray, shape (n_trs, n_voxels, n_subs)
    The target voxel data for all subjects
    obs_seed_ts : ndarray, shape (n_trs, 1, n_subs)
    The observed seed timeseries for all subjects
    rng : np.random.RandomState
    Random state for reproducibility

    Returns:
    --------
    loo_isfc : ndarray, shape (n_voxels, n_subs)
    LOO ISFC values for each subject at each voxel
    loo_z : ndarray, shape (n_voxels, n_subs)
    Fisher z-transformed LOO ISFC values
    """
    n_trs, n_voxels, n_subs = group_data.shape
    target_sum = np.sum(group_data, axis=2)  # (n_trs, n_voxels)

    loo_isfc = np.zeros((n_voxels, n_subs), dtype=np.float32)

    for s in range(n_subs):
        # Phase randomize only subject s's seed timeseries
        # obs_seed_ts shape is (n_trs, 1, n_subs), get subject s: (n_trs, 1)
        subj_seed = obs_seed_ts[:, 0, s]  # (n_trs,)
        shifted_seed = phase_randomize(subj_seed.reshape(-1, 1), random_state=rng).flatten()

        # Compute mean target of OTHER subjects (exclude subject s)
        others_target_mean = (target_sum - group_data[:, :, s]) / (n_subs - 1)  # (n_trs, n_voxels)

        # Correlate shifted seed with unshifted others' target mean
        # Demean both
        shifted_seed_dm = shifted_seed - np.mean(shifted_seed)
        others_target_dm = others_target_mean - np.mean(others_target_mean, axis=0, keepdims=True)

        # Compute correlation for each voxel
        # shifted_seed_dm: (n_trs,), others_target_dm: (n_trs, n_voxels)
        numerator = np.sum(shifted_seed_dm[:, np.newaxis] * others_target_dm, axis=0)
        denom_seed = np.sqrt(np.sum(shifted_seed_dm ** 2))
        denom_target = np.sqrt(np.sum(others_target_dm ** 2, axis=0))

        with np.errstate(divide='ignore', invalid='ignore'):
            r = numerator / (denom_seed * denom_target)
            r = np.nan_to_num(r, nan=0.0)

        loo_isfc[:, s] = r

    # Fisher z-transform
    loo_isfc_clipped = np.clip(loo_isfc, -0.99999, 0.99999)
    loo_z = np.arctanh(loo_isfc_clipped)

    return loo_isfc, loo_z


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
    parser.add_argument('--chunk_size', type=int, default=20000,
                        help=f'Chunk size (default: {config.CHUNK_SIZE})')
    parser.add_argument("--isfc_method", type=str, choices=["loo", "pairwise"], default="loo",
                        help="ISFC method for phaseshift: 'loo' (leave-one-out) or 'pairwise'")
    return parser.parse_args()

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

def _run_phaseshift_iter(i, obs_seed_ts, group_data_path, chunk_size, use_tfce, mask, tfce_E, tfce_H, seed,pairwise=False):

    # Generate surrogate seed by phase randomizing the OBSERVED seed
    group_data=np.load(group_data_path,mmap_mode="r")
    rng = np.random.RandomState(seed)

    if pairwise:
        # PAIRWISE MODE: Phase randomize the entire seed timeseries, then compute pairwise ISFC
        # This is the original approach - valid for pairwise correlations
        surr_seed_ts = phase_randomize(obs_seed_ts, voxelwise=False, random_state=rng)
        surr_raw, surr_z = run_isfc_computation(group_data, surr_seed_ts, pairwise=True, chunk_size=chunk_size)
    else:
        # LOO MODE: Use correct approach - phase randomize only the left-out subject's seed
        # and compare to unshifted target mean of others
        surr_raw, surr_z = _compute_loo_null_isfc(group_data, obs_seed_ts, rng)
 
    null_mean = np.nanmean(surr_z, axis=1)  # Mean over subjects
 
    if use_tfce:
        # FWER Correction: Return max statistic
        null_mean_3d = np.zeros(mask.shape, dtype=np.float32)
        null_mean_3d[mask] = null_mean
        null_mean_3d = apply_tfce(null_mean_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
        return np.max(np.abs(null_mean_3d))
    else:
        return null_mean

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
    results = Parallel(n_jobs=1, verbose=5)(
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


def run_phaseshift(condition, roi_id, seed_coords, seed_radius, n_perms, data_dir, mask_file,output_dir, chunk_size=config.CHUNK_SIZE,seed_file=None, use_tfce=False, tfce_E=0.5, tfce_H=2.0,checkpoint_every=25,resume=False,pairwise=False):
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
    # Derive method string from pairwise parameter (not from path)

    isfc_method_str = "pairwise" if pairwise else "loo"
    print(f"Running Phase Shift (n={n_perms}, chunk_size={chunk_size}, pairwise={pairwise})...")

    mask, affine = load_mask(mask_file, roi_id=roi_id)
    if np.sum(mask) == 0: raise ValueError("Empty mask")

    # group_data = load_data(condition, config.SUBJECTS, mask, data_dir)
    group_data_path = os.path.join(output_dir, f"group_data_{condition}.npy")
    print(group_data_path)

    if not os.path.exists(group_data_path):
        group_data = load_data(condition, config.SUBJECTS, mask, data_dir)
        if group_data is None:
            raise ValueError("No data")
        np.save(group_data_path, group_data)
    else:
        group_data = np.load(group_data_path)

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
    obs_isfc_raw, obs_isfc_z = run_isfc_computation(group_data, obs_seed_ts, pairwise=pairwise, chunk_size=chunk_size)
    obs_mean_z = np.nanmean(obs_isfc_z, axis=1) # (V,)
    n_voxels = int(obs_mean_z.shape[0])

    if use_tfce:
        # Apply TFCE to observed map
        obs_mean_z_3d = np.zeros(mask.shape, dtype=np.float32)
        obs_mean_z_3d[mask] = obs_mean_z
        obs_mean_z_3d = apply_tfce(obs_mean_z_3d, mask, E=tfce_E, H=tfce_H, two_sided=True)
        obs_mean_z = obs_mean_z_3d[mask]
    
    cluster=SLURMCluster(account='menon',
    queue='menon,owners,normal',
    processes=1,
    cores=1,
    memory='64GB',
    local_directory='/scratch/users/daelsaid/',
    walltime='10:00:00',
    death_timeout=600,
    job_script_prologue=["export PATH=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/isc_env/bin:$PATH",
    f"export PYTHONPATH={TI_CODE_DIR}:$PYTHONPATH",
    "export OMP_NUM_THREADS=1",
    "export MKL_NUM_THREADS=1",
    "export OPENBLAS_NUM_THREADS=1",
    "export NUMEXPR_NUM_THREADS=1",
    "export VECLIB_MAXIMUM_THREADS=1"])
    cluster.scale(jobs=8)

    #cluster = LocalCluster(n_workers=1, threads_per_worker=8)
    client=Client(cluster)
    client.upload_file('../isfc/config.py')
    client.upload_file('../isfc/pipeline_utils_dist.py')
    client.wait_for_workers(2)
    client.run(lambda: __import__("os").environ.get("SLURM_MEM_PER_NODE", "no_slurm_mem"))

    if output_dir is None:
        raise ValueError("output_dir is required for checkpointing and resume")
 
    # Use isfc_method_str derived from pairwise parameter (already defined above)
    ckpt_name = f"isfc_{condition}_phaseshift_{isfc_method_str}"
    run_tag = ckpt_name
 
    nullmaps_dir = os.path.join(output_dir, f"null_maps_{isfc_method_str}_{condition}", run_tag)
    print(f"Null maps directory: {nullmaps_dir}")
    os.makedirs(nullmaps_dir, exist_ok=True)


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
        abs_obs = np.abs(obs_mean_z).astype(np.float32)

    expected_meta = {
        "condition": condition,
        "roi_id": roi_id,
        "chunk_size": int(chunk_size),
        "use_tfce": bool(use_tfce),
        "tfce_E": float(tfce_E),
        "tfce_H": float(tfce_H),
        "n_voxels": int(n_voxels),
        "n_perms": int(n_perms),
        "seed_coords": tuple(seed_coords) if seed_coords is not None else None,
        "seed_radius": float(seed_radius) if seed_radius is not None else None,
        "seed_file": os.path.abspath(seed_file) if seed_file else None,
        "pairwise":bool(pairwise)

    }

    if resume and os.path.exists(ckpt_path):
        ck = np.load(ckpt_path, allow_pickle=True)
        completed = int(ck["completed"])
        meta = ck["meta"].item()

        if meta != expected_meta:
            raise ValueError(f"Checkpoint meta mismatch.\nFound: {meta}\nExpected: {expected_meta}")

        if use_tfce:
            null_max_stats = ck["null_max_stats"].astype(np.float32).tolist()
        else:
            count_greater = ck["count_greater"].astype(np.int32)

        print(f"Resuming from checkpoint {ckpt_path}", flush=True)
        print(f"Completed permutations: {completed} / {n_perms}", flush=True)
    print(f"Starting {n_perms} permutations", flush=True)

    for b0 in range(completed, n_perms, checkpoint_every):
        b1 = min(b0 + checkpoint_every, n_perms)
        print(f"Running permutations {b0} to {b1 - 1}", flush=True)

        tasks = [
            dask_delayed(_run_phaseshift_iter)(
                i,
                obs_seed_ts,
                group_data_path,
                chunk_size,
                use_tfce,
                mask,
                tfce_E,
                tfce_H,
                1000 + i,
                pairwise
            )
            for i in range(b0, b1)
        ]

        futures = client.compute(tasks)
        progress(futures)
        batch_results = client.gather(futures)

        if use_tfce:
            null_max_stats.extend([float(x) for x in batch_results])
        else:
            for null_mean in batch_results:
                null_mean = null_mean.astype(np.float32)
                count_greater += (np.abs(null_mean) >= abs_obs).astype(np.int32)

        completed = b1

        if use_tfce:
            np.savez(
                ckpt_path,
                completed=completed,
                null_max_stats=np.array(null_max_stats, dtype=np.float32),
                meta=expected_meta,
            )
        else:
            np.savez(
                ckpt_path,
                completed=completed,
                count_greater=count_greater,
                meta=expected_meta,
            )

        print(f"Saved checkpoint at {completed}: {ckpt_path}", flush=True)

    # compute p values from checkpoint accumulators
    if use_tfce:
        null_max_arr = np.array(null_max_stats, dtype=np.float32)
        p_values = (
            (np.sum(null_max_arr[np.newaxis, :] >= np.abs(obs_mean_z[:, np.newaxis]), axis=1) + 1)
            / (len(null_max_arr) + 1)
        ).astype(np.float32)
    else:
        p_values = ((count_greater + 1) / (completed + 1)).astype(np.float32)

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
    isfc_method = args.isfc_method
    pairwise = (isfc_method == "pairwise")
 
    print(f"--- Step 2: ISFC Statistics ---")
    print(f"Method: {method}")
    print(f"ISFC Method: {isfc_method} (pairwise={pairwise})")
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
             
        # mean_map, p_values, mask_data, mask_affine = run_phaseshift(
        #     args.condition, args.roi_id, seed_coords, args.seed_radius, args.n_perms, 
        #     data_dir=data_dir, mask_file=mask_file, output_dir=output_dir, chunk_size=chunk_size, seed_file=args.seed_file,use_tfce=args.use_tfce, tfce_E=args.tfce_E, tfce_H=args.tfce_H,checkpoint_every=args.checkpoint_every,resume=args.resume)
        # output_dir = os.path.join(args.output_dir,args.condition,isfc_method)
        mean_map, p_values, mask_data, mask_affine = run_phaseshift(
            args.condition, args.roi_id, seed_coords, args.seed_radius, args.n_perms,data_dir=data_dir, mask_file=mask_file, output_dir=output_dir, chunk_size=chunk_size,
            seed_file=args.seed_file, use_tfce=args.use_tfce, tfce_E=args.tfce_E, tfce_H=args.tfce_H,checkpoint_every=args.checkpoint_every, resume=args.resume,pairwise=pairwise)

        # Include isfc_method in base_name to prevent overwriting between loo and pairwise runs
        base_name = f"isfc_{args.condition}_{isfc_method}_{method}{seed_suffix}"

        
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
