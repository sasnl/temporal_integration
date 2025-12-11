import os
import glob
import numpy as np
import nibabel as nib
from brainiak.isc import isfc
import time
import gc

# Configuration
DATA_DIR = '/Users/tongshan/Documents/TemporalIntegration/data/td/hpf'
OUTPUT_DIR = '/Users/tongshan/Documents/TemporalIntegration/result'
# Using same mask file as template/default
MASK_FILE = '/Users/tongshan/Documents/TemporalIntegration/code/ISCtoolbox_v3_R340/templates/HarvardOxford-cort-maxprob-thr25-2mm.nii'
# Set ROI_ID to a specific integer to run ISFC only on that region
# If None, runs on valid voxels in mask (WARNING: Whole brain ISFC is huge!)
ROI_ID = 48  # Defaulting to an example ROI to prevent accidental whole-brain explosion on first run.

# -------------------------------------------------
# CONDITIONS = ['TI1_orig', 'TI1_sent', 'TI1_word']
CONDITIONS = ['TI1_orig', 'TI1_sent', 'TI1_word']
SUBJECTS = ['11051', '12501', '12503', '12505', '12506', '12515', '12516', '12517', '12527', '12530', '12532', '12538', '12542', '9409']

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
    n_voxels = data_list[0].shape[1]
    n_subs = len(data_list)
    
    group_data = np.zeros((min_tr, n_voxels, n_subs), dtype=np.float32)
    for i, d in enumerate(data_list):
        group_data[:, :, i] = d[:min_tr, :]
        
    return group_data

def run_isfc(data):
    """
    Compute ISFC (Inter-Subject Functional Correlation).
    
    Parameters
    ----------
    data : array_like (TRs, Voxels, Subjects)
    
    Returns
    -------
    isfc_matrix : ndarray (Voxels, Voxels)
        The group-average leave-one-out ISFC matrix.
    """
    print("Running ISFC analysis...")
    # brainiak.isc.isfc computes the ISFC.
    # pairwise=False: computes leave-one-out ISFCs (one map per subject).
    # summary_statistic='mean': averages these maps into a single group matrix.
    isfc_matrix = isfc(data, pairwise=False, summary_statistic='mean', vectorize_isfcs=False)
    
    return isfc_matrix

def save_matrix(matrix, output_path):
    print(f"Saving ISFC matrix to {output_path}")
    print(f"  Matrix shape: {matrix.shape}")
    np.save(output_path, matrix)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Note: ROI_ID is critical here for ISFC to avoid massive matrices if not intended
    mask, affine = load_mask(MASK_FILE, roi_id=ROI_ID)
    
    if np.sum(mask) == 0:
        print("Error: Mask is empty. Exiting.")
        return
        
    n_voxels = np.sum(mask)
    expected_size_mb = (n_voxels * n_voxels * 4) / (1024 * 1024)
    print(f"Expected ISFC matrix size: {n_voxels} x {n_voxels} (approx {expected_size_mb:.2f} MB)")
    
    if expected_size_mb > 2000:
        print("WARNING: Matrix size is > 2GB. Ensure you have enough RAM and disk space.")

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
        
        isfc_result = run_isfc(group_data)
        
        # Modify output filename
        suffix = f"_roi{ROI_ID}" if ROI_ID is not None else "_wholebrain"
        output_filename = f"isfc_{condition}{suffix}.npy"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        save_matrix(isfc_result, output_path)
        
        print(f"Finished {condition} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
