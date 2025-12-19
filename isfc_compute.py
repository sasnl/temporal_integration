import os
import argparse
import numpy as np
import time
from brainiak.isc import isfc
from joblib import Parallel, delayed
from isc_utils import load_mask, load_data, save_map, save_plot, get_seed_mask, load_seed_data

# Configuration
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/td/hpf'
OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/ISCtoolbox_v3_R340/templates/MNI152_T1_2mm_brain_mask.nii'
CHUNK_SIZE = 5000
SUBJECTS = ['11051', '12501', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']

def parse_args():
    parser = argparse.ArgumentParser(description='Step 1: Compute ISFC Maps (Raw and Fisher-Z)')
    parser.add_argument('--condition', type=str, required=True, 
                        help='Condition name (e.g., TI1_orig)')
    parser.add_argument('--method', type=str, choices=['loo', 'pairwise'], default='loo',
                        help='ISFC method: "loo" (Leave-One-Out) or "pairwise"')
    parser.add_argument('--roi_id', type=int, default=None,
                        help='Optional: ROI ID to mask (default: Whole Brain)')
    parser.add_argument('--seed_x', type=float, required=True, help='Seed X coordinate (MNI)')
    parser.add_argument('--seed_y', type=float, required=True, help='Seed Y coordinate (MNI)')
    parser.add_argument('--seed_z', type=float, required=True, help='Seed Z coordinate (MNI)')
    parser.add_argument('--seed_radius', type=float, default=5, help='Seed Radius in mm (default: 5)')
    return parser.parse_args()

def compute_isfc_chunk(target_chunk, seed_ts, pairwise):
    """
    Computes ISFC between seed_ts and target_chunk.
    
    target_chunk: (TR, V_Chunk, n_subjects)
    seed_ts: (TR, 1, n_subjects)
    pairwise: bool
    
    Returns: 
    - if pairwise=False (LOO): (n_subjects, 1, V_Chunk) -> squeeze to (n_subjects, V_Chunk)
    - if pairwise=True: (n_pairs, 1, V_Chunk) -> squeeze to (n_pairs, V_Chunk)
    """
    # brainiak.isc.isfc computes correlation between dataset A and dataset B
    # If pairwise=False, it does Leave-One-Out ISFC:
    #   Corr(Subject_i_Seed, Mean_Others_Target)
    #   Wait, BrainIAK ISFC(loo) computes Corr(Subject_i_A, Mean_Others_B) AND Corr(Subject_i_B, Mean_Others_A) usually? 
    #   Actually brainiak.isc.isfc with summary_statistic=None returns the vector of Correlations.
    #   Let's check documentation logic:
    #   ISFC(data, targets) where data=(TR, V1, S), targets=(TR, V2, S).
    #   LOO: For each subject S_i, computes corr between S_i[data] and Mean(Others[targets])
    #   AND S_i[targets] and Mean(Others[data]). 
    #   It returns average of these two? Or just one?
    #   Actually for Seed-based ISFC, we want: Corr(Seed_Subject_i, Target_Voxels_Others_Mean).
    #   Symmetric ISFC (Seed <-> Target) is exactly what brainiak.isc.isfc does.
    
    res = isfc(seed_ts, target_chunk, pairwise=pairwise, summary_statistic=None, vectorize_isfcs=False)
    # res shape: (n_samples, 1, V_Chunk)
    
    return res.squeeze(axis=1) # (n_samples, V_Chunk)

def run_isfc_computation(data, seed_ts, pairwise=False):
    """
    Run ISFC computation in parallel chunks.
    """
    print(f"Running ISFC computation (Pairwise={pairwise})")
    n_trs, n_voxels, n_subs = data.shape
    
    n_chunks = int(np.ceil(n_voxels / CHUNK_SIZE))
    chunks = []
    for i in range(n_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, n_voxels)
        chunks.append(data[:, start_idx:end_idx, :])
        
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_isfc_chunk)(chunk, seed_ts, pairwise) for chunk in chunks
    )
    
    # Reassemble
    sample_dim = results[0].shape[0]
    isfc_maps = np.zeros((n_voxels, sample_dim), dtype=np.float32)
    
    for i, res in enumerate(results):
        start_idx = i * CHUNK_SIZE
        end_idx = min((i + 1) * CHUNK_SIZE, n_voxels)
        isfc_maps[start_idx:end_idx, :] = res.T
        
    return isfc_maps

def main():
    args = parse_args()
    condition = args.condition
    method = args.method
    roi_id = args.roi_id
    seed_coords = (args.seed_x, args.seed_y, args.seed_z)
    seed_radius = args.seed_radius
    
    pairwise = (method == 'pairwise')
    
    print(f"--- Step 1: ISFC Computation ---")
    print(f"Condition: {condition}")
    print(f"Method: {method}")
    print(f"ROI: {roi_id if roi_id else 'Whole Mask'}")
    print(f"Seed: {seed_coords} (r={seed_radius}mm)")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load Mask
    mask, affine = load_mask(MASK_FILE, roi_id=roi_id)
    if np.sum(mask) == 0: return

    # Load Data
    group_data = load_data(condition, SUBJECTS, mask, DATA_DIR)
    if group_data is None: return
    
    start_time = time.time()
    
    # Prepare Seed
    seed_mask = get_seed_mask(mask.shape, affine, seed_coords, seed_radius)
    seed_ts = load_seed_data(group_data, seed_mask, mask) # (TR, 1, S)
    print(f"Seed timecourse loaded: {seed_ts.shape}")
    
    # Compute Raw ISFC
    isfc_raw = run_isfc_computation(group_data, seed_ts, pairwise=pairwise)
    
    # Fischer Z
    isfc_raw_clipped = np.clip(isfc_raw, -0.99999, 0.99999)
    isfc_z = np.arctanh(isfc_raw_clipped)
    
    # Save Maps
    roi_suffix = f"_roi{roi_id}" if roi_id is not None else ""
    seed_suffix = f"_seed{int(seed_coords[0])}_{int(seed_coords[1])}_{int(seed_coords[2])}_r{int(seed_radius)}"
    base_name = f"isfc_{condition}_{method}{seed_suffix}{roi_suffix}"
    
    raw_path = os.path.join(OUTPUT_DIR, f"{base_name}_desc-raw.nii.gz")
    z_path = os.path.join(OUTPUT_DIR, f"{base_name}_desc-zscore.nii.gz")
    
    save_map(isfc_raw, mask, affine, raw_path)
    save_map(isfc_z, mask, affine, z_path)
    
    # Save Mean Plot
    mean_z = np.nanmean(isfc_z, axis=1)
    plot_path = os.path.join(OUTPUT_DIR, f"{base_name}_desc-meanz.png")
    # Temp save mean z for plotting
    temp_path = os.path.join(OUTPUT_DIR, f"temp_{base_name}.nii.gz")
    save_map(mean_z, mask, affine, temp_path)
    save_plot(temp_path, plot_path, f"Mean ISFC (Z) {condition} {seed_coords}")
    os.remove(temp_path)
    
    print(f"Computation finished in {time.time() - start_time:.2f} seconds.")
    print(f"Outputs:\n  {raw_path}\n  {z_path}")

if __name__ == "__main__":
    main()
