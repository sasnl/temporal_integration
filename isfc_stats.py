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

# Configuration
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/td/hpf'
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/ISCtoolbox_v3_R340/templates/MNI152_T1_2mm_brain_mask.nii'
SUBJECTS = ['11051', '12501', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']
CHUNK_SIZE = 5000

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
    parser.add_argument('--output_dir', type=str, default='/Users/tongshan/Documents/TemporalIntegration/result',
                        help='Output directory')
    return parser.parse_args()

def run_ttest(data_4d):
    """
    Run one-sample T-test against 0 on the last dimension of data.
    """
    print("Running T-test...")
    t_stats, p_values = ttest_1samp(data_4d, popmean=0, axis=-1, nan_policy='omit')
    mean_map = np.nanmean(data_4d, axis=-1)
    
    # One-sided check (Positive correlation)
    # p_values from ttest_1samp are two-sided. 
    # For one-sided > 0:
    # if t > 0: p_one_sided = p_two_sided / 2
    # if t < 0: p_one_sided = 1 - (p_two_sided / 2) 
    # Let's stick to standard 2-sided or user preference. Usually ISC we care about positive.
    # For simplicity here, returning two-sided p-values to be conservative, 
    # but marking mask where mean > 0 if we want directional.
    
    return mean_map, p_values

def run_bootstrap(data_4d, n_bootstraps=1000, random_state=42):
    """
    Run bootstrap on 4D map (subjects dimension).
    """
    print(f"Running Bootstrap (n={n_bootstraps})...")
    n_voxels, n_samples = data_4d.shape
    rng = np.random.RandomState(random_state)
    
    observed_mean = np.nanmean(data_4d, axis=1) # (V,)
    
    # Center data to test null hypothesis mean=0
    # Actually for "significance of correlation", we usually test against 0.
    # Standard bootstrap for one-sample test:
    # 1. Resample data with replacement -> calculate mean*.
    # 2. Build distribution of mean*.
    # 3. Confidence Interval? 
    # Or Hypothesis test:
    # To test H0: mu=0.
    # We can use bootstrap to estimate distribution of (Mean - 0).
    # Wait, simple bootstrap hypothesis testing:
    # Center data so mean is 0 -> data_centered = data - observed_mean
    # Resample data_centered -> mean*_null.
    # calc p = prop(abs(mean*_null) >= abs(observed_mean))
    
    data_centered = data_4d - observed_mean[:, np.newaxis]
    null_means = np.zeros((n_voxels, n_bootstraps), dtype=np.float32)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, n_samples, size=n_samples)
        sample = data_centered[:, indices]
        null_means[:, i] = np.nanmean(sample, axis=1)
        
    # P-value (Two-sided)
    # count how many null means are more extreme than observed mean
    # Since we centered, we compare to |observed_mean| vs |null_mean|? 
    # Wait, observed statistic is 'observed_mean'.
    # null distribution is centered at 0.
    # p = sum(abs(null_means) >= abs(observed_mean)) / N
    
    with np.errstate(invalid='ignore'):
         p_values = np.sum(np.abs(null_means) >= np.abs(observed_mean[:, np.newaxis]), axis=1) / (n_bootstraps + 1)
    
    return observed_mean, p_values

def run_phaseshift(condition, roi_id, seed_coords, seed_radius, n_perms):
    """
    Run Phase Shift randomization.
    1. Load Data
    2. Extract Seed
    3. Generate surrogate seeds
    4. Compute ISFC for observed +/- surrogates
    """
    print(f"Running Phase Shift (n={n_perms})...")
    
    mask, affine = load_mask(MASK_FILE, roi_id=roi_id)
    if np.sum(mask) == 0: raise ValueError("Empty mask")
    
    group_data = load_data(condition, SUBJECTS, mask, DATA_DIR)
    if group_data is None: raise ValueError("No data")
    
    seed_mask = get_seed_mask(mask.shape, affine, seed_coords, seed_radius)
    obs_seed_ts = load_seed_data(group_data, seed_mask, mask) # (TR, 1, S)
    
    print("  Generating surrogate seeds...")
    # Generate surrogates
    # We need to run ISFC for observed seed and N surrogate seeds.
    # To save time, we can run them all together if memory permits.
    # BrainIAK ISFC takes (TR, V, S).
    # We need:
    # 1. Obs ISFC: Corr(ObsSeed, Target) (LOO)
    # 2. Null ISFC: Corr(SurrSeed, Target) (LOO)
    
    # Let's use a loop or stacking.
    # Stacking seeds: (TR, N_Perms+1, S) ? computing ISFC against target (TR, V, S)
    # brainiak.isc.isfc doesn't broadcast well for multiple seeds against one target unless we loop.
    
    # We will reuse run_isfc_computation logic but need to inject different seeds.
    # But run_isfc_computation expects single seed. 
    # We should modify it or loop here. looping is safer.
    
    # 1. Observed
    print("  Computing Observed ISFC...")
    obs_isfc_raw = run_isfc_computation(group_data, obs_seed_ts, pairwise=False) # (V, S)
    obs_isfc_z = np.arctanh(np.clip(obs_isfc_raw, -0.99999, 0.99999))
    obs_mean_z = np.nanmean(obs_isfc_z, axis=1) # (V,)
    
    # 2. Null Distribution
    null_means = np.zeros((obs_mean_z.shape[0], n_perms), dtype=np.float32)
    
    for i in range(n_perms):
        # Generate surrogate seed
        # phase_randomize expects (TR, V, S)
        # We randomize the seed timecourses
        surr_seed_ts = phase_randomize(obs_seed_ts, voxelwise=False, random_state=i+1000)
        
        # Compute ISFC
        # For efficiency, we really should optimize this loop if N is large.
        # But for now, we use the function.
        # To avoid printing too much, we could suppress print or modify run_isfc_computation.
        # We'll just call it.
        if i % 10 == 0: print(f"  Permutation {i+1}/{n_perms}")
        
        surr_raw = run_isfc_computation(group_data, surr_seed_ts, pairwise=False)
        surr_z = np.arctanh(np.clip(surr_raw, -0.99999, 0.99999))
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
    
    print(f"--- Step 2: ISFC Statistics ---")
    print(f"Method: {method}")
    print(f"Threshold: {threshold}")
    
    mask_affine = None
    mask_data = None
    mean_map = None
    p_values = None
    
    if args.roi_id is not None: 
         # Load mask to ensure we have it if needed for phaseshift or map masking
         mask_data, mask_affine = load_mask(MASK_FILE, roi_id=args.roi_id)

    if method == 'phaseshift':
        if not args.condition or args.seed_x is None:
            raise ValueError("Phaseshift requires --condition and --seed coordinates")
            
        seed_coords = (args.seed_x, args.seed_y, args.seed_z)
        mean_map, p_values, mask_data, mask_affine = run_phaseshift(
            args.condition, args.roi_id, seed_coords, args.seed_radius, args.n_perms
        )
        base_name = f"isfc_{args.condition}_{method}"
        
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
             mask_data, _ = load_mask(MASK_FILE, roi_id=None)
             
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
