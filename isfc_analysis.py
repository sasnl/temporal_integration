import os
import glob
import numpy as np
import nibabel as nib
from brainiak.isc import isfc, phase_randomize
import time
import gc
import argparse
import sys
from nilearn import plotting
import matplotlib.pyplot as plt
import warnings
from joblib import Parallel, delayed

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in true_divide")

# Configuration
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/td/hpf'
OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/ISCtoolbox_v3_R340/templates/MNI152_T1_2mm_brain_mask.nii'
ROI_ID = None 

# Seed-based ISFC Configuration
SEED_COORD = (0, -52, 26)  # PMC
SEED_RADIUS = 5  # Radius in mm
CHUNK_SIZE = 5000 # Voxels per joblib chunk

# -------------------------------------------------
CONDITIONS = ['TI1_orig']
SUBJECTS = ['11012', '11036', '11051', '12501', '12502', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12531', '12532', '12538', '12542', '9409']

def parse_args():
    parser = argparse.ArgumentParser(description='Run Optimized ISFC analysis (BrainIAK)')
    parser.add_argument('--n_perms', type=int, default=1000,
                        help='Number of permutations (default: 1000)')
    parser.add_argument('--roi_id', type=int, default=ROI_ID,
                        help='ROI ID')
    parser.add_argument('--condition', type=str, default=None,
                        help='Specific condition')
    return parser.parse_args()

def load_mask(mask_path, roi_id=None):
    print(f"Loading mask from {mask_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    mask_img = nib.load(mask_path)
    data = mask_img.get_fdata()
    
    if roi_id is not None:
        print(f"  Selecting ROI with ID: {roi_id}")
        mask_data = np.isclose(data, roi_id)
        if np.sum(mask_data) == 0:
            print(f"  WARNING: ROI {roi_id} not found.")
    else:
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
        # print(f"  Loading {os.path.basename(file_path)}")
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32)
        masked_data = data[mask].T
        data_list.append(masked_data)
        
    if not data_list: return None
    
    n_trs = [d.shape[0] for d in data_list]
    min_tr = min(n_trs)
    n_voxels = data_list[0].shape[1]
    n_subs = len(data_list)
    
    group_data = np.zeros((min_tr, n_voxels, n_subs), dtype=np.float32)
    for i, d in enumerate(data_list):
        group_data[:, :, i] = d[:min_tr, :]
        
    return group_data

def coord_to_voxel(affine, coords):
    inv_affine = np.linalg.inv(affine)
    coords_h = np.array(list(coords) + [1])
    voxel_coords = inv_affine @ coords_h
    return np.round(voxel_coords[:3]).astype(int)

def get_seed_mask(mask_shape, affine, center_coords, radius_mm):
    print(f"Creating spherical seed mask at {center_coords} with r={radius_mm}mm")
    center_vox = coord_to_voxel(affine, center_coords)
    vox_sizes = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    min_vox_size = np.min(vox_sizes)
    voxel_radius = int(np.ceil(radius_mm / min_vox_size)) + 1
    
    nx, ny, nz = mask_shape
    min_x = max(0, center_vox[0] - voxel_radius)
    max_x = min(nx, center_vox[0] + voxel_radius + 1)
    min_y = max(0, center_vox[1] - voxel_radius)
    max_y = min(ny, center_vox[1] + voxel_radius + 1)
    min_z = max(0, center_vox[2] - voxel_radius)
    max_z = min(nz, center_vox[2] + voxel_radius + 1)
    
    seed_mask = np.zeros(mask_shape, dtype=bool)
    count = 0
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            for z in range(min_z, max_z):
                pt_vox = np.array([x, y, z, 1])
                pt_world = affine @ pt_vox
                dist = np.linalg.norm(pt_world[:3] - np.array(center_coords))
                if dist <= radius_mm:
                    seed_mask[x, y, z] = True
                    count += 1
    print(f"  Seed mask contains {count} voxels")
    return seed_mask

def load_seed_data(group_data, seed_mask, whole_brain_mask):
    valid_seed_mask = seed_mask & whole_brain_mask
    if np.sum(valid_seed_mask) == 0: raise ValueError("Seed mask not in analysis mask")
    mapping = np.full(whole_brain_mask.shape, -1, dtype=int)
    mapping[whole_brain_mask] = np.arange(np.sum(whole_brain_mask))
    seed_indices = mapping[valid_seed_mask]
    seed_voxels = group_data[:, seed_indices, :]
    seed_ts = np.mean(seed_voxels, axis=1, keepdims=True) # (TR, 1, Subj)
    return seed_ts

def process_chunk_brainiak(target_chunk, all_seeds):
    """
    Computes ISFC using BrainIAK for all seeds against target_chunk.
    
    target_chunk: (TR, V_Chunk, Subj)
    all_seeds: (N_Seeds, TR, 1, Subj) [0=Observed, 1..=Surrogates]
    """
    n_seeds = all_seeds.shape[0]
    n_vox = target_chunk.shape[1]
    
    results = np.zeros((n_seeds, n_vox), dtype=np.float32)
    
    # BrainIAK ISFC: compute ISFC between two sets of data
    # Unfortunately brainiak.isc.isfc doesn't support 1-vs-Many directly in a vectorized way for N_seeds.
    # We must loop through seeds.
    
    for i in range(n_seeds):
        # seed_ts: (TR, 1, Subj)
        seed_ts = all_seeds[i]
        
        # brainiak.isc.isfc(data, targets, ...)
        # Computes ISFC between data and targets.
        # Returns (n_vox_data, n_vox_targets) matrix if vectorized=False (default behavior is matrix)
        # Here data is 1 voxel, targets is V_Chunk.
        # result shape: (1, V_Chunk)
        
        # bias correction: res usually (n_seed_vox, n_target_vox). If seed is 1 voxel, might be (n_target_vox,)
        res = isfc(seed_ts, target_chunk, pairwise=False, summary_statistic='mean', vectorize_isfcs=False)
        
        if res.ndim == 1:
            results[i, :] = res
        else:
            results[i, :] = res[0, :]
        
    return results

def main():
    args = parse_args()
    n_perms = args.n_perms
    mask_roi_id = args.roi_id
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print(f"--- MODE: SEED-BASED ISFC (BrainIAK Parallel) ---")
    print(f"Seed: {SEED_COORD}, Perms: {n_perms}")

    mask, affine = load_mask(MASK_FILE, roi_id=mask_roi_id)
    if np.sum(mask) == 0: return
    n_voxels = np.sum(mask)

    conditions_to_run = [args.condition] if args.condition else CONDITIONS

    for condition in conditions_to_run:
        print(f"\nProcessing {condition}...")
        start_time = time.time()
        
        group_data = load_data(condition, SUBJECTS, mask, DATA_DIR)
        if group_data is None: continue
        
        print(f"  Data loaded: {group_data.shape}. Generating seeds...")
        
        # 1. Prepare Seeds
        seed_mask = get_seed_mask(mask.shape, affine, SEED_COORD, SEED_RADIUS)
        obs_seed = load_seed_data(group_data, seed_mask, mask) # (TR, 1, S)
        
        # Generate Surrogates
        all_seeds_list = [obs_seed]
        if n_perms > 0:
            print(f"  Generating {n_perms} phase-randomized surrogate seeds...")
            for i in range(n_perms):
                surr = phase_randomize(obs_seed, voxelwise=False, random_state=i+1000)
                all_seeds_list.append(surr)
                
        all_seeds = np.array(all_seeds_list) # (N_Perms+1, TR, 1, S)
        print(f"  Seeds prepared: {all_seeds.shape}")
        
        # 2. Run Parallel Analysis on Chunks
        print(f"  Running Parallel Analysis ({n_voxels} voxels, ChunkSize={CHUNK_SIZE})...")
        
        n_chunks = int(np.ceil(n_voxels / CHUNK_SIZE))
        chunks = []
        for i in range(n_chunks):
            s = i * CHUNK_SIZE
            e = min((i+1) * CHUNK_SIZE, n_voxels)
            chunks.append(group_data[:, s:e, :])
            
        # Run Jobs
        results = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_chunk_brainiak)(chunk, all_seeds) for chunk in chunks
        )
        
        # 3. Assemble Results
        print("  Assembling results...")
        full_results = np.hstack(results) # (N_Seeds, Voxels)
        
        obs_isfc_map = full_results[0, :]
        
        if n_perms > 0:
            null_dist = full_results[1:, :] # (N_Perms, Voxels)
            
            # Calculate P-values (Two-sided)
            count = np.sum(np.abs(null_dist) >= np.abs(obs_isfc_map), axis=0)
            p_values = (count + 1) / (n_perms + 1)
        else:
            p_values = None
            
        # 4. Save Outputs
        print("  Saving outputs...")
        suffix = f"_r{SEED_RADIUS}_{condition}" 
        
        # NIfTI
        out_name = f"isfc_seed{SEED_COORD[0]}_{SEED_COORD[1]}_{SEED_COORD[2]}{suffix}.nii.gz"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        save_nifti(obs_isfc_map, MASK_FILE, out_path)
        save_plot(out_path, out_path.replace('.nii.gz', '.png'), f"Seed ISFC {condition}")
        
        if p_values is not None:
             p_name = f"isfc_seed{SEED_COORD[0]}_{SEED_COORD[1]}_{SEED_COORD[2]}{suffix}_pvals.nii.gz"
             p_path = os.path.join(OUTPUT_DIR, p_name)
             save_nifti(p_values, MASK_FILE, p_path)
             
             # Significant Map
             sig_map = obs_isfc_map.copy()
             sig_map[p_values >= 0.05] = 0
             s_name = f"isfc_seed{SEED_COORD[0]}_{SEED_COORD[1]}_{SEED_COORD[2]}{suffix}_sig05.nii.gz"
             s_path = os.path.join(OUTPUT_DIR, s_name)
             save_nifti(sig_map, MASK_FILE, s_path)
             save_plot(s_path, s_path.replace('.nii.gz', '.png'), f"Sig Seed ISFC (p<0.05) {condition}")
             
        print(f"Finished {condition} in {time.time() - start_time:.1f}s")

def save_nifti(vector_data, mask_file, output_path):
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata().astype(bool)
    out_data = np.zeros(mask_data.shape, dtype=np.float32)
    out_data[mask_data] = vector_data.flatten()
    out_img = nib.Nifti1Image(out_data, mask_img.affine, mask_img.header)
    nib.save(out_img, output_path)

def save_plot(nifti_path, output_image_path, title):
    try:
        display = plotting.plot_stat_map(nifti_path, title=title, display_mode='z', cut_coords=8, colorbar=True)
        display.savefig(output_image_path)
        display.close()
    except Exception as e:
        print(f"Plot failed: {e}")

if __name__ == "__main__":
    main()
