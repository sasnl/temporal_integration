import os
import glob
import numpy as np
import nibabel as nib
from brainiak.isc import isc, bootstrap_isc, phaseshift_isc
from nilearn import plotting
import matplotlib.pyplot as plt
import time
import gc

# Configuration
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/td/hpf'
OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/ISCtoolbox_v3_R340/templates/MNI152_T1_2mm_brain_mask.nii'

CONDITION = 'TI1_orig'
SUBJECTS = ['11051', '12501', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']
CHUNK_SIZE = 5000
N_PERMS = 100

def load_mask(mask_path):
    print(f"Loading mask from {mask_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at: {mask_path}")
    mask_img = nib.load(mask_path)
    data = mask_img.get_fdata()
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
            continue
        
        file_path = files[0]
        print(f"  Loading {os.path.basename(file_path)}")
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32)
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

from joblib import Parallel, delayed

def process_chunk(chunk_data, n_perms):
    # 1. Bootstrap (Random Effects)
    chunk_isc = isc(chunk_data, pairwise=False)
    b_obs, _, b_p, _ = bootstrap_isc(chunk_isc, pairwise=False, n_bootstraps=n_perms, random_state=42)
    
    # 2. Phaseshift (Fixed Effects)
    # Returns: observed, p, distribution
    p_obs, p_p, _ = phaseshift_isc(chunk_data, pairwise=False, n_shifts=n_perms, random_state=42)
    return b_obs, b_p, p_obs, p_p

def run_comparison_chunked(data, chunk_size=CHUNK_SIZE):
    print(f"Running Comparison analysis in PARALLEL chunks (Size: {chunk_size}, n_jobs=-1)...")
    n_trs, n_voxels, n_subs = data.shape
    
    n_chunks = int(np.ceil(n_voxels / chunk_size))
    
    # Prepare chunk generator
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        chunks.append(data[:, start_idx:end_idx, :])

    # Run parallel
    # joblib handles numpy array masking efficiently
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_chunk)(chunk, N_PERMS) for chunk in chunks
    )
    
    # Reassemble
    boot_obs_map = np.zeros(n_voxels, dtype=np.float32)
    boot_p_map = np.zeros(n_voxels, dtype=np.float32)
    phase_obs_map = np.zeros(n_voxels, dtype=np.float32)
    phase_p_map = np.zeros(n_voxels, dtype=np.float32)
    
    for i, (b_obs, b_p, p_obs, p_p) in enumerate(results):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        
        boot_obs_map[start_idx:end_idx] = b_obs
        boot_p_map[start_idx:end_idx] = b_p
        phase_obs_map[start_idx:end_idx] = p_obs
        phase_p_map[start_idx:end_idx] = p_p
            
    return boot_obs_map, boot_p_map, phase_obs_map, phase_p_map

def save_map(data_1d, mask, affine, output_path):
    print(f"Saving map to {output_path}")
    vol_data = np.zeros(mask.shape, dtype=np.float32)
    vol_data[mask] = data_1d
    img = nib.Nifti1Image(vol_data, affine)
    nib.save(img, output_path)
    return output_path

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    mask, affine = load_mask(MASK_FILE)
    
    # Load Data
    start_time = time.time()
    group_data = load_data(CONDITION, SUBJECTS, mask, DATA_DIR)
    if group_data is None:
        print("No data found.")
        return
    print(f"Data loaded in {time.time() - start_time:.2f}s. Shape: {group_data.shape}")
    
    # Run Comparison
    t0 = time.time()
    boot_obs, boot_p, phase_obs, phase_p = run_comparison_chunked(group_data)
    print(f"Comparison finished in {time.time() - t0:.2f}s")
    
    # Save Maps
    save_map(boot_p, mask, affine, os.path.join(OUTPUT_DIR, f"isc_{CONDITION}_bootstrap_p.nii.gz"))
    file_boot_sig = save_map(boot_obs * (boot_p < 0.05), mask, affine, os.path.join(OUTPUT_DIR, f"isc_{CONDITION}_bootstrap_sig.nii.gz"))

    save_map(phase_p, mask, affine, os.path.join(OUTPUT_DIR, f"isc_{CONDITION}_phaseshift_p.nii.gz"))
    file_phase_sig = save_map(phase_obs * (phase_p < 0.05), mask, affine, os.path.join(OUTPUT_DIR, f"isc_{CONDITION}_phaseshift_sig.nii.gz"))
    
    # Visualization Comparison
    print("\nGenerating Comparison Plot...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    plotting.plot_stat_map(file_boot_sig, display_mode='z', cut_coords=8, axes=axes[0], colorbar=True, title=f"Bootstrap ISC (Random Effects) p<0.05")
    plotting.plot_stat_map(file_phase_sig, display_mode='z', cut_coords=8, axes=axes[1], colorbar=True, title=f"Phaseshift ISC (Fixed Effects) p<0.05")
    
    output_plot = os.path.join(OUTPUT_DIR, f"isc_comparison_{CONDITION}.png")
    plt.savefig(output_plot)
    print(f"Comparison plot saved to {output_plot}")

if __name__ == "__main__":
    main()

