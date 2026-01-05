import os
import argparse
import numpy as np
import time
from brainiak.isc import isc
from joblib import Parallel, delayed
from isc_utils import load_mask, load_data, save_map, save_plot
import config

def parse_args():
    parser = argparse.ArgumentParser(description='Step 1: Compute ISC Maps (Raw and Fisher-Z)')
    parser.add_argument('--condition', type=str, required=True, 
                        help='Condition name (e.g., TI1_orig)')
    parser.add_argument('--method', type=str, choices=['loo', 'pairwise'], default='loo',
                        help='ISC method: "loo" (Leave-One-Out) or "pairwise"')
    parser.add_argument('--roi_id', type=int, default=None,
                        help='Optional: ROI ID to mask (default: Whole Brain)')
    
    # Configurable Paths
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to input data (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help=f'Path to output directory (default: {config.OUTPUT_DIR})')
    parser.add_argument('--mask_file', type=str, default=config.MASK_FILE,
                        help=f'Path to mask file (default: {config.MASK_FILE})')
    return parser.parse_args()

def compute_isc_chunk(chunk_data, pairwise):
    """
    Compute ISC for a single chunk of data.
    """
    # isc() returns shape (n_subjects, n_voxels) for pairwise=False (LOO)
    # isc() returns shape (n_pairs, n_voxels) for pairwise=True
    # Note: brainiak.isc.isc default is pairwise=False (Leave-one-out)
    return isc(chunk_data, pairwise=pairwise)

def run_isc_computation(data, pairwise=False):
    """
    Run ISC computation in parallel chunks.
    Returns: isc_maps (n_voxels, n_samples) where n_samples is n_subjects (LOO) or n_pairs.
    """
    print(f"Running ISC computation (Pairwise={pairwise})")
    n_trs, n_voxels, n_subs = data.shape
    
    n_chunks = int(np.ceil(n_voxels / config.CHUNK_SIZE))
    chunks = []
    for i in range(n_chunks):
        start_idx = i * config.CHUNK_SIZE
        end_idx = min((i + 1) * config.CHUNK_SIZE, n_voxels)
        chunks.append(data[:, start_idx:end_idx, :])

    # Parallel execution
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_isc_chunk)(chunk, pairwise) for chunk in chunks
    )
    
    # Reassemble
    # Determine output shape from first result
    # Result shape from isc is (n_samples, n_chunk_voxels)
    sample_dim = results[0].shape[0]
    
    isc_maps = np.zeros((n_voxels, sample_dim), dtype=np.float32)
    
    for i, res in enumerate(results):
        start_idx = i * config.CHUNK_SIZE
        end_idx = min((i + 1) * config.CHUNK_SIZE, n_voxels)
        
        # Transpose to match (n_voxels, n_samples) for easier slicing later
        isc_maps[start_idx:end_idx, :] = res.T
        
    return isc_maps

def main():
    args = parse_args()
    condition = args.condition
    method = args.method
    roi_id = args.roi_id
    
    # Path Args
    data_dir = args.data_dir
    output_dir = args.output_dir
    mask_file = args.mask_file
    
    pairwise = (method == 'pairwise')
    
    print(f"--- Step 1: ISC Computation ---")
    print(f"Condition: {condition}")
    print(f"Method: {method}")
    print(f"ROI: {roi_id if roi_id else 'Whole Mask'}")
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
        print("Error: No data loaded.")
        return
    
    start_time = time.time()
    
    # Compuete Raw ISC
    # Shape: (n_voxels, n_samples)
    isc_raw = run_isc_computation(group_data, pairwise=pairwise)
    
    # Compute Fisher-Z
    # Clip to avoid infinity
    isc_raw_clipped = np.clip(isc_raw, -0.99999, 0.99999)
    isc_z = np.arctanh(isc_raw_clipped)
    
    # Prepare Filenames
    roi_suffix = f"_roi{roi_id}" if roi_id is not None else ""
    base_name = f"isc_{condition}_{method}{roi_suffix}"
    
    raw_path = os.path.join(output_dir, f"{base_name}_desc-raw.nii.gz")
    z_path = os.path.join(output_dir, f"{base_name}_desc-zscore.nii.gz")
    
    # Save 4D Maps
    save_map(isc_raw, mask, affine, raw_path)
    save_map(isc_z, mask, affine, z_path)
    
    # Save Mean Maps for Quick Check (3D)
    mean_raw = np.nanmean(isc_raw, axis=1)
    mean_z = np.nanmean(isc_z, axis=1) # Mean of Z-scores
    
    mean_raw_path = os.path.join(output_dir, f"{base_name}_desc-meanraw.nii.gz")
    mean_z_path = os.path.join(output_dir, f"{base_name}_desc-meanz.nii.gz")
    
    save_map(mean_raw, mask, affine, mean_raw_path)
    save_map(mean_z, mask, affine, mean_z_path)
    
    # Save Plot of Mean Z
    plot_path = os.path.join(output_dir, f"{base_name}_desc-meanz.png")
    save_plot(mean_z_path, plot_path, f"Mean ISC (Fisher-Z) - {condition} - {method}")
    
    print(f"Computation finished in {time.time() - start_time:.2f} seconds.")
    print(f"Outputs:\n  {raw_path}\n  {z_path}")

if __name__ == "__main__":
    main()
