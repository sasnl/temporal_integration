import os
import glob
import numpy as np
import nibabel as nib
from brainiak.isc import isc, bootstrap_isc, phaseshift_isc
import time
import gc
from nilearn import plotting
import matplotlib.pyplot as plt
import argparse
from joblib import Parallel, delayed

# Configuration
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/td/hpf'
OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/ISCtoolbox_v3_R340/templates/MNI152_T1_2mm_brain_mask.nii'

# Update conditions to only include TI1_orig
CONDITIONS = ['TI1_orig']
SUBJECTS = ['11051', '12501', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']

CHUNK_SIZE = 5000  # Number of voxels to process at a time

def parse_args():
    parser = argparse.ArgumentParser(description='Run ISC analysis (Bootstrap or Phaseshift)')
    parser.add_argument('--method', type=str, choices=['bootstrap', 'phaseshift'], default='bootstrap',
                        help='Statistical method: "bootstrap" (Random Effects) or "phaseshift" (Fixed Effects)')
    parser.add_argument('--n_perms', type=int, default=1000,
                        help='Number of permutations/bootstraps (default: 1000)')
    parser.add_argument('--roi_id', type=int, default=None,
                        help='Optional: ROI integer ID to mask specific region (default: Whole Brain)')
    return parser.parse_args()

def load_mask(mask_path, roi_id=None):
    print(f"Loading mask from {mask_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at: {mask_path}")
    mask_img = nib.load(mask_path)
    data = mask_img.get_fdata()
    
    if roi_id is not None:
        print(f"  Selecting ROI with ID: {roi_id}")
        mask_data = np.isclose(data, roi_id)
        if np.sum(mask_data) == 0:
            print(f"  WARNING: ROI {roi_id} not found in mask (0 voxels).")
    else:
        # Default behavior: defined anywhere > 0
        mask_data = data > 0
        
    print(f"  Mask contains {np.sum(mask_data)} voxels")
    return mask_data, mask_img.affine

def load_data(condition, subjects, mask, data_dir):
    print(f"Loading data for condition: {condition}")
    data_list = []
    
    cond_dir = os.path.join(data_dir, condition)
    
    for sub in subjects:
        search_pattern = os.path.join(cond_dir, f"{sub}_*.nii")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"Warning: No file found for subject {sub} in {condition}")
            continue
        
        file_path = files[0]
        print(f"  Loading {os.path.basename(file_path)}")
        
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32)
        
        # Apply mask
        masked_data = data[mask].T
        data_list.append(masked_data)
        
    if not data_list:
        return None
    
    n_trs = [d.shape[0] for d in data_list]
    min_tr = min(n_trs)
    
    n_voxels = data_list[0].shape[1]
    n_subs = len(data_list)
    
    group_data = np.zeros((min_tr, n_voxels, n_subs), dtype=np.float32)
    for i, d in enumerate(data_list):
        group_data[:, :, i] = d[:min_tr, :]
        
    return group_data

def process_chunk(chunk_data, method, n_perms):
    # Helper to run analysis on a single chunk (joblib worker)
    if method == 'bootstrap':
        chunk_isc = isc(chunk_data, pairwise=False)
        # bootstrap_isc returns: observed, ci, p, distribution
        observed, _, p, _ = bootstrap_isc(chunk_isc, pairwise=False, n_bootstraps=n_perms, random_state=42)
    elif method == 'phaseshift':
        # phaseshift_isc returns: observed, p, distribution
        observed, p, _ = phaseshift_isc(chunk_data, pairwise=False, n_shifts=n_perms, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    return observed, p

def run_isc_parallel(data, method, n_perms, chunk_size=CHUNK_SIZE):
    print(f"Running {method} analysis in PARALLEL chunks (Size: {chunk_size}, n_jobs=-1)...")
    n_trs, n_voxels, n_subs = data.shape
    
    n_chunks = int(np.ceil(n_voxels / chunk_size))
    
    # Prepare chunks generator
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        chunks.append(data[:, start_idx:end_idx, :])

    # Execute parallel
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_chunk)(chunk, method, n_perms) for chunk in chunks
    )
    
    # Reassemble maps
    mean_isc_map = np.zeros(n_voxels, dtype=np.float32)
    p_value_map = np.zeros(n_voxels, dtype=np.float32)
    
    for i, (observed, p) in enumerate(results):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        
        mean_isc_map[start_idx:end_idx] = observed
        p_value_map[start_idx:end_idx] = p
            
    return mean_isc_map, p_value_map

def save_map(data_1d, mask, affine, output_path):
    print(f"Saving map to {output_path}")
    vol_data = np.zeros(mask.shape, dtype=np.float32)
    vol_data[mask] = data_1d
    
    img = nib.Nifti1Image(vol_data, affine)
    nib.save(img, output_path)
    return output_path

def save_plot(nifti_path, output_image_path, title):
    print(f"Generating plot to {output_image_path}")
    display = plotting.plot_stat_map(nifti_path, title=title, display_mode='z', cut_coords=8, colorbar=True)
    display.savefig(output_image_path)
    display.close()

def main():
    args = parse_args()
    method = args.method
    n_perms = args.n_perms
    roi_id = args.roi_id
    
    print(f"--- ISC Analysis Config ---")
    print(f"Method: {method}")
    print(f"Permutations: {n_perms}")
    print(f"ROI ID: {roi_id if roi_id else 'Whole Brain'}")
    print(f"---------------------------")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    mask, affine = load_mask(MASK_FILE, roi_id=roi_id)
    
    if np.sum(mask) == 0:
        print("Error: Mask is empty. Exiting.")
        return

    for condition in CONDITIONS:
        print(f"\nProcessing {condition}...")
        start_time = time.time()
        
        try:
            del group_data
        except NameError:
            pass
        gc.collect()
        
        group_data = load_data(condition, SUBJECTS, mask, DATA_DIR)
        
        if group_data is None:
            print(f"Skipping {condition} due to missing data.")
            continue
            
        print(f"  Data shape: {group_data.shape}")
        
        # Run Analysis
        mean_isc, p_values = run_isc_parallel(group_data, method, n_perms)
        
        # Filename suffix based on method
        method_suffix = f"_{method}"
        roi_suffix = f"_roi{roi_id}" if roi_id is not None else ""
        
        # Save raw Mean ISC map
        output_filename = f"isc_{condition}{method_suffix}{roi_suffix}_mean.nii.gz"
        save_map(mean_isc, mask, affine, os.path.join(OUTPUT_DIR, output_filename))
        
        # Save P-value map
        p_filename = f"isc_{condition}{method_suffix}{roi_suffix}_pvalues.nii.gz"
        save_map(p_values, mask, affine, os.path.join(OUTPUT_DIR, p_filename))
        
        # Create Significant Map (p < 0.05)
        sig_indices = p_values < 0.05
        sig_isc = np.zeros_like(mean_isc)
        sig_isc[sig_indices] = mean_isc[sig_indices]
        
        sig_filename = f"isc_{condition}{method_suffix}{roi_suffix}_significant_p05.nii.gz"
        sig_path = os.path.join(OUTPUT_DIR, sig_filename)
        save_map(sig_isc, mask, affine, sig_path)
        
        # Plotting
        plot_filename = f"isc_{condition}{method_suffix}{roi_suffix}_significant_p05.png"
        plot_path = os.path.join(OUTPUT_DIR, plot_filename)
        save_plot(sig_path, plot_path, f"Significant ISC ({method}, p<0.05) - {condition}")
        
        print(f"Finished {condition} in {time.time() - start_time:.2f} seconds")
        print(f"Outputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

