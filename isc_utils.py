import os
import glob
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt

def load_mask(mask_path, roi_id=None):
    """
    Load mask file and optionally select a specific ROI.
    """
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
    """
    Load fMRI data for a list of subjects and a specific condition, applying a mask.
    Returns: group_data (n_TRs, n_voxels, n_subjects)
    """
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
        # print(f"  Loading {os.path.basename(file_path)}")
        
        img = nib.load(file_path)
        data = img.get_fdata(dtype=np.float32)
        
        # Apply mask
        masked_data = data[mask].T
        masked_data = np.nan_to_num(masked_data)
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

def save_map(data, mask, affine, output_path):
    """
    Save specific data (1D, 3D, or 4D) back to a Nifti file using the mask geometry.
    If data is 4D (e.g. n_voxels x n_maps), it will be saved as a 4D nifti.
    """
    print(f"Saving map to {output_path}")
    
    if data.ndim == 1:
        # 3D Volume case (single map)
        vol_data = np.zeros(mask.shape, dtype=np.float32)
        vol_data[mask] = data
    elif data.ndim == 2:
        # 4D Volume case (multiple maps, e.g. per subject)
        n_maps = data.shape[1]
        # Initialize 4D array: X, Y, Z, T
        vol_shape = mask.shape + (n_maps,)
        vol_data = np.zeros(vol_shape, dtype=np.float32)
        
        # Fill each volume
        for i in range(n_maps):
            temp_vol = np.zeros(mask.shape, dtype=np.float32)
            temp_vol[mask] = data[:, i]
            vol_data[..., i] = temp_vol
            
    else:
        raise ValueError(f"Unsupported data dimension for saving: {data.ndim}")
    
    img = nib.Nifti1Image(vol_data, affine)
    nib.save(img, output_path)
    return output_path

def save_plot(nifti_path, output_image_path, title):
    """
    Generate and save a static plot of a nifti map.
    """
    print(f"Generating plot to {output_image_path}")
    # Note: plot_stat_map works best with 3D images. 
    # If 4D is passed, we might need to mean it or pick first volume, 
    # but usually this function is called on summary maps (mean/p-value) which are 3D.
    try:
        display = plotting.plot_stat_map(nifti_path, title=title, display_mode='z', cut_coords=8, colorbar=True)
        display.savefig(output_image_path)
        display.close()
    except Exception as e:
        print(f"Failed to generate plot: {e}")
