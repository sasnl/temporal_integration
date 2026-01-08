import os
import argparse
import numpy as np
import time
from brainiak.isc import isfc
from joblib import Parallel, delayed
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from pipeline_utils import load_mask, load_data, save_map, save_plot, get_seed_mask, load_seed_data
import config

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
    
    # Configurable Paths
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to input data (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help=f'Path to output directory (default: {config.OUTPUT_DIR})')
    parser.add_argument('--mask_file', type=str, default=config.MASK_FILE,
                        help=f'Path to mask file (default: {config.MASK_FILE})')
    parser.add_argument('--chunk_size', type=int, default=config.CHUNK_SIZE,
                        help=f'Chunk size (default: {config.CHUNK_SIZE})')
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
    
    isfc_raw = res.squeeze(axis=1) # (n_samples, V_Chunk)
    
    # Fischer Z transformation for the chunk
    isfc_raw_clipped = np.clip(isfc_raw, -0.99999, 0.99999)
    isfc_z = np.arctanh(isfc_raw_clipped)
    
    return isfc_raw, isfc_z

def run_isfc_computation(data, seed_ts, pairwise=False, chunk_size=config.CHUNK_SIZE):
    """
    Run ISFC computation in parallel chunks.
    data: (TRs, Voxels, Subjects)
    seed_ts: (TRs, 1, Subjects)
    """
    print(f"Running ISFC computation (Pairwise={pairwise}, Chunk Size: {chunk_size})")
    n_trs, n_voxels, n_subs = data.shape
    
    n_chunks = int(np.ceil(n_voxels / chunk_size))
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        chunks.append(data[:, start_idx:end_idx, :])
        
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_isfc_chunk)(chunk, seed_ts, pairwise) for chunk in chunks
    )
    
    # Reassemble
    # result shape from compute_isfc_chunk is (isfc_raw_chunk, isfc_z_chunk)
    # where each chunk is (n_samples, V_Chunk)
    sample_dim = results[0][0].shape[0] # Get n_samples from the first raw chunk
    
    isfc_raw = np.zeros((n_voxels, sample_dim), dtype=np.float32)
    isfc_z = np.zeros((n_voxels, sample_dim), dtype=np.float32)
    
    for i, (r_chunk, z_chunk) in enumerate(results):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        isfc_raw[start_idx:end_idx, :] = r_chunk.T
        isfc_z[start_idx:end_idx, :] = z_chunk.T
        
    return isfc_raw, isfc_z

def main():
    args = parse_args()
    condition = args.condition
    method = args.method
    roi_id = args.roi_id
    seed_coords = (args.seed_x, args.seed_y, args.seed_z)
    seed_radius = args.seed_radius
    
    data_dir = args.data_dir
    output_dir = args.output_dir
    mask_file = args.mask_file
    chunk_size = args.chunk_size
    
    pairwise = (method == 'pairwise')
    
    print(f"--- Step 1: ISFC Computation ---")
    print(f"Condition: {condition}")
    print(f"Method: {method}")
    print(f"ROI: {roi_id if roi_id else 'Whole Mask'}")
    print(f"Seed: {seed_coords} (r={seed_radius}mm)")
    print(f"Chunk Size: {chunk_size}")
    print(f"Data Dir: {data_dir}")
    print(f"Output Dir: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Load Mask
    mask, affine = load_mask(mask_file, roi_id=roi_id)
    if np.sum(mask) == 0: 
        print("Error: Empty mask.")
        return

    # Load Data
    group_data = load_data(condition, config.SUBJECTS, mask, data_dir)
    if group_data is None: 
        print("Error: Failed to load data.")
        return

    start_time = time.time()
    
    # Prepare Seed
    seed_mask = get_seed_mask(mask.shape, affine, seed_coords, seed_radius)
    seed_ts = load_seed_data(group_data, seed_mask, mask) # (TR, 1, S)
    print(f"Seed timecourse loaded: {seed_ts.shape}")
    
    # Compute Raw and Z ISFC
    isfc_raw, isfc_z = run_isfc_computation(group_data, seed_ts, pairwise=pairwise, chunk_size=chunk_size)
    
    # Fischer Z (Already computed in run_isfc_computation)

    
    # Save Maps
    roi_suffix = f"_roi{roi_id}" if roi_id is not None else ""
    seed_suffix = f"_seed{int(seed_coords[0])}_{int(seed_coords[1])}_{int(seed_coords[2])}_r{int(seed_radius)}"
    base_name = f"isfc_{condition}_{method}{seed_suffix}{roi_suffix}"
    
    raw_path = os.path.join(output_dir, f"{base_name}_desc-raw.nii.gz")
    z_path = os.path.join(output_dir, f"{base_name}_desc-zscore.nii.gz")
    
    save_map(isfc_raw, mask, affine, raw_path)
    save_map(isfc_z, mask, affine, z_path)
    
    # Save Mean Plot
    mean_z = np.nanmean(isfc_z, axis=1)
    plot_path = os.path.join(output_dir, f"{base_name}_desc-meanz.png")
    # Temp save mean z for plotting
    temp_path = os.path.join(output_dir, f"temp_{base_name}.nii.gz")
    save_map(mean_z, mask, affine, temp_path)
    save_plot(temp_path, plot_path, f"Mean ISFC (Z) {condition} {seed_coords}")
    os.remove(temp_path)
    
    print(f"Computation finished in {time.time() - start_time:.2f} seconds.")
    print(f"Outputs:\n  {raw_path}\n  {z_path}")

if __name__ == "__main__":
    main()
