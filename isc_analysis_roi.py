import os
import glob
import numpy as np
import nibabel as nib
from brainiak.isc import isc
import time
import gc

# Configuration
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/td/hpf'
OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'

# ----------------- ROI SELECTION -----------------
# Option 1: Whole Brain / Custom Binary Mask (DEFAULT)
# Data is selected wherever values > 0
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/ISCtoolbox_v3_R340/templates/MNI152_T1_2mm_brain_mask.nii'
ROI_ID = None  # Set to None to use the whole mask

# Option 2: Atlas-based ROI (Example)
# Uncomment the lines below to use a specific region from an atlas
# MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/ISCtoolbox_v3_R340/templates/HarvardOxford-cort-maxprob-thr25-2mm.nii'
# ROI_ID = 48  # Replace with the integer ID of your desired region
# -------------------------------------------------

CONDITIONS = ['TI1_orig', 'TI1_sent', 'TI1_word']
SUBJECTS = ['11051', '12501', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']

CHUNK_SIZE = 5000  # Number of voxels to process at a time

def load_mask(mask_path, roi_id=None):
    print(f"Loading mask from {mask_path}")
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found at: {mask_path}")
    mask_img = nib.load(mask_path)
    data = mask_img.get_fdata()
    
    if roi_id is not None:
        print(f"  Selecting ROI with ID: {roi_id}")
        # Create binary mask for just this specific integer
        mask_data = np.isclose(data, roi_id)
        if np.sum(mask_data) == 0:
            print(f"  WARNING: ROI {roi_id} not found in mask (0 voxels). Please check your ID.")
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
        # Find file for subject
        search_pattern = os.path.join(cond_dir, f"{sub}_*.nii")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"Warning: No file found for subject {sub} in {condition}")
            continue
        
        file_path = files[0]
        print(f"  Loading {os.path.basename(file_path)}")
        
        img = nib.load(file_path)
        # Load as float32 to save memory
        data = img.get_fdata(dtype=np.float32)
        
        # Apply mask
        masked_data = data[mask].T
        data_list.append(masked_data)
        
    if not data_list:
        return None
    
    # Check TRs
    n_trs = [d.shape[0] for d in data_list]
    min_tr = min(n_trs)
    if len(set(n_trs)) > 1:
        print(f"Warning: Mismatch in TRs. Truncating to {min_tr}")
        
    # Stack data: (TRs, Voxels, Subjects)
    # Pre-allocate array to avoid memory spike during stacking
    n_voxels = data_list[0].shape[1]
    n_subs = len(data_list)
    
    group_data = np.zeros((min_tr, n_voxels, n_subs), dtype=np.float32)
    for i, d in enumerate(data_list):
        group_data[:, :, i] = d[:min_tr, :]
        
    return group_data

def run_isc_chunked(data, chunk_size=CHUNK_SIZE):
    print("Running ISC analysis in chunks...")
    n_trs, n_voxels, n_subs = data.shape
    mean_isc_map = np.zeros(n_voxels, dtype=np.float32)
    
    n_chunks = int(np.ceil(n_voxels / chunk_size))
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        
        # Extract chunk: (TRs, ChunkVoxels, Subjects)
        chunk_data = data[:, start_idx:end_idx, :]
        
        # Run ISC on chunk
        # pairwise=False returns (n_voxels, n_subjects) - LOO ISC
        chunk_isc = isc(chunk_data, pairwise=False)
        
        # Compute mean across subjects
        # chunk_isc shape is (n_subjects, n_voxels)
        chunk_mean_isc = np.mean(chunk_isc, axis=0)
        
        mean_isc_map[start_idx:end_idx] = chunk_mean_isc
        
        if (i + 1) % 10 == 0:
            print(f"  Processed chunk {i + 1}/{n_chunks}")
            
    return mean_isc_map

def save_map(data_1d, mask, affine, output_path):
    print(f"Saving map to {output_path}")
    vol_data = np.zeros(mask.shape, dtype=np.float32)
    vol_data[mask] = data_1d
    
    img = nib.Nifti1Image(vol_data, affine)
    nib.save(img, output_path)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    mask, affine = load_mask(MASK_FILE, roi_id=ROI_ID)
    
    if np.sum(mask) == 0:
        print("Error: Mask is empty. Exiting.")
        return

    for condition in CONDITIONS:
        print(f"\nProcessing {condition}...")
        start_time = time.time()
        
        # Explicitly delete previous data to free memory
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
        
        isc_result = run_isc_chunked(group_data)
        
        # Modify output filename if using ROI
        suffix = f"_roi{ROI_ID}" if ROI_ID is not None else ""
        output_filename = f"isc_{condition}{suffix}.nii.gz"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        save_map(isc_result, mask, affine, output_path)
        
        print(f"Finished {condition} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
