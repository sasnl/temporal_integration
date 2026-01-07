import os
import argparse
import numpy as np
import nibabel as nib
from scipy.stats import ttest_1samp
from brainiak.isc import phase_randomize
from joblib import Parallel, delayed
from isc_utils import load_mask, load_data, save_map, save_plot, get_seed_mask, load_seed_data
from isfc_compute import run_isfc_computation 
# Import run_isfc_computation to reuse logic for phase shift re-computation
import config

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
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='P-value threshold (default: 0.05)')
    parser.add_argument('--seed_x', type=float, help='Seed X (Required for Phase Shift)')
    parser.add_argument('--seed_y', type=float, help='Seed Y (Required for Phase Shift)')
    parser.add_argument('--seed_z', type=float, help='Seed Z (Required for Phase Shift)')
    parser.add_argument('--seed_radius', type=float, default=5, help='Seed Radius (Required for Phase Shift)')
    
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

# ... (skipping unchanged code) ...

def run_phaseshift(condition, roi_id, seed_coords, seed_radius, n_perms, data_dir, mask_file, chunk_size=config.CHUNK_SIZE):
    """
    Run Phase Shift randomization.
    """
    print(f"Running Phase Shift (n={n_perms}, chunk_size={chunk_size})...")
    
    mask, affine = load_mask(mask_file, roi_id=roi_id)
    # ... (loading logic unchanged) ...
    if np.sum(mask) == 0: raise ValueError("Empty mask")
    group_data = load_data(condition, config.SUBJECTS, mask, data_dir)
    if group_data is None: raise ValueError("No data")
    
    seed_mask = get_seed_mask(mask.shape, affine, seed_coords, seed_radius)
    obs_seed_ts = load_seed_data(group_data, seed_mask, mask)
    
    print("  Generating surrogate seeds...")
    
    # 1. Observed
    print("  Computing Observed ISFC...")
    obs_isfc_raw, obs_isfc_z = run_isfc_computation(group_data, obs_seed_ts, pairwise=False, chunk_size=chunk_size)
    obs_mean_z = np.nanmean(obs_isfc_z, axis=1) # (V,)
    
    # 2. Null Distribution
    null_means = np.zeros((obs_mean_z.shape[0], n_perms), dtype=np.float32)
    
    for i in range(n_perms):
        # Generate surrogate seed
        surr_seed_ts = phase_randomize(obs_seed_ts, voxelwise=False, random_state=i+1000)
        
        if i % 10 == 0: print(f"  Permutation {i+1}/{n_perms}")
        
        surr_raw, surr_z = run_isfc_computation(group_data, surr_seed_ts, pairwise=False, chunk_size=chunk_size)
        null_means[:, i] = np.nanmean(surr_z, axis=1) # Mean over subjects
        
    # P-values
    # Two-sided
    count = np.sum(np.abs(null_means) >= np.abs(obs_mean_z[:, np.newaxis]), axis=1)
    p_values = (count + 1) / (n_perms + 1)
    
    return obs_mean_z, p_values, mask, affine

def main():
    args = parse_args()
    method = args.method
    roi_id = args.roi_id
    threshold = args.threshold
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
        if not args.condition or args.seed_x is None:
            raise ValueError("Phaseshift requires --condition and --seed coordinates")
            
        seed_coords = (args.seed_x, args.seed_y, args.seed_z)
        mean_map, p_values, mask_data, mask_affine = run_phaseshift(
            args.condition, args.roi_id, seed_coords, args.seed_radius, args.n_perms, 
            data_dir=data_dir, mask_file=mask_file, chunk_size=chunk_size
        )
        seed_suffix = f"_seed{int(seed_coords[0])}_{int(seed_coords[1])}_{int(seed_coords[2])}_r{int(args.seed_radius)}"
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
            mean_vals, p_vals_vec = run_ttest(masked_data)
        elif method == 'bootstrap':
            mean_vals, p_vals_vec = run_bootstrap(masked_data, n_bootstraps=args.n_perms)
            
        # Reconstruct maps
        mean_map = np.zeros(mask_data.shape, dtype=np.float32)
        mean_map[mask_data] = mean_vals
        
        p_values = np.ones(mask_data.shape, dtype=np.float32)
        p_values[mask_data] = p_vals_vec
        
        input_base = os.path.basename(args.input_map).replace('.nii.gz', '').replace('_desc-zscore', '').replace('_desc-raw', '')
        base_name = f"{input_base}_{method}"

    # Save Outputs
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # 1. P-values
    p_path = os.path.join(output_dir, f"{base_name}_desc-pvalues.nii.gz")
    save_map(p_values, mask_data, mask_affine, p_path)
    
    # 2. Significant Map
    sig_map = mean_map.copy()
    sig_map[p_values >= threshold] = 0
    # Also mask out 0s if they were 0 originally
    
    sig_path = os.path.join(output_dir, f"{base_name}_desc-sig_p{str(threshold).replace('.', '')}.nii.gz")
    save_map(sig_map, mask_data, mask_affine, sig_path)
    
    # 3. Plot
    plot_path = os.path.join(output_dir, f"{base_name}_desc-sig.png")
    save_plot(sig_path, plot_path, f"Sig ISFC ({method}, p<{threshold})")
    
    print("Done")
    print(f"Outputs:\n  {sig_path}")

if __name__ == "__main__":
    main()
