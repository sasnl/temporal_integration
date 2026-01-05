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

def coord_to_voxel(affine, coords):
    """
    Convert world coordinates to voxel coordinates.
    """
    inv_affine = np.linalg.inv(affine)
    coords_h = np.array(list(coords) + [1])
    voxel_coords = inv_affine @ coords_h
    return np.round(voxel_coords[:3]).astype(int)

def get_seed_mask(mask_shape, affine, center_coords, radius_mm):
    """
    Create a spherical seed mask within the volume.
    """
    print(f"Creating spherical seed mask at {center_coords} with r={radius_mm}mm")
    center_vox = coord_to_voxel(affine, center_coords)
    vox_sizes = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    min_vox_size = np.min(vox_sizes)
    # Calculate radius in voxels
    voxel_radius = int(np.ceil(radius_mm / min_vox_size)) + 1
    
    nx, ny, nz = mask_shape
    # Bounding box
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
    """
    Extract seed timecourse from group data.
    group_data: (TR, V_Whole, S)
    seed_mask: 3D boolean mask of seed
    whole_brain_mask: 3D boolean mask of analysis
    Returns: seed_ts (TR, 1, S)
    """
    # Find intersection of seed and data mask
    valid_seed_mask = seed_mask & whole_brain_mask
    if np.sum(valid_seed_mask) == 0:
        raise ValueError("Seed mask does not overlap with analysis mask.")
        
    # Mapping from 3D space to 1D index in group_data
    # whole_brain_mask defines the 1D indices [0, 1, ..., V_Whole-1]
    mapping = np.full(whole_brain_mask.shape, -1, dtype=int)
    mapping[whole_brain_mask] = np.arange(np.sum(whole_brain_mask))
    
    seed_indices = mapping[valid_seed_mask]
    
    # Check if any indices are -1 (shouldn't happen due to logic above)
    if np.any(seed_indices == -1):
         raise ValueError("Error mapping seed indices.")
         
    seed_voxels = group_data[:, seed_indices, :] # (TR, V_Seed, S)
    seed_ts = np.mean(seed_voxels, axis=1, keepdims=True) # (TR, 1, S)
    return seed_ts

def save_map(data, mask, affine, output_path):
    """
    Save specific data (1D, 3D, or 4D) back to a Nifti file using the mask geometry.
    If data is 4D (e.g. n_voxels x n_maps), it will be saved as a 4D nifti.
    """
    print(f"Saving map to {output_path}")
    
    if data.ndim == 1:
        # 1D Vector case (masked data) -> 3D Volume
        vol_data = np.zeros(mask.shape, dtype=np.float32)
        vol_data[mask] = data
    elif data.ndim == 3:
        # 3D Volume case (already shaped)
        # Verify shape matches mask
        if data.shape != mask.shape:
             raise ValueError(f"Data shape {data.shape} does not match mask shape {mask.shape}")
        vol_data = data
    elif data.ndim == 2:
        # 2D Matrix (Voxels x Time/Samples) -> 4D Volume
        # Check if first dim matches mask sum (Vector case) or mask shape (impossible if 2D)
        # Actually in compute script we pass (n_voxels, n_samples).
        n_maps = data.shape[1]
        vol_shape = mask.shape + (n_maps,)
        vol_data = np.zeros(vol_shape, dtype=np.float32)
        
        # Fill each volume
        for i in range(n_maps):
            temp_vol = np.zeros(mask.shape, dtype=np.float32)
            # Assuming data is (MaskedVoxels, N)
            temp_vol[mask] = data[:, i]
            vol_data[..., i] = temp_vol
            
    else:
        raise ValueError(f"Unsupported data dimension for saving: {data.ndim}")
    
    img = nib.Nifti1Image(vol_data, affine)
    nib.save(img, output_path)
    return output_path

def save_plot(nifti_path, output_image_path, title, dpi=300, transparent=True, positive_only=False):
    """
    Generate and save a static plot of a nifti map.
    """
    print(f"Generating plot to {output_image_path}")
    # Note: plot_stat_map works best with 3D images. 
    # If 4D is passed, we might need to mean it or pick first volume, 
    # but usually this function is called on summary maps (mean/p-value) which are 3D.
    try:
        img_to_plot = nifti_path
        if positive_only:
            img = nib.load(nifti_path)
            data = img.get_fdata()
            # Mask negative values to NaN (better than 0 for transparency/thresholding)
            data[data < 0] = np.nan
            img_to_plot = nib.Nifti1Image(data, img.affine)

        # Use black_bg=True for better contrast (standard in fMRI)
        # If positive_only, disable symmetric colorbar so it doesn't show negative range
        symmetric_cbar = not positive_only
        cmap = 'hot' if positive_only else 'cold_hot'
        
        display = plotting.plot_stat_map(
            img_to_plot, 
            title=title, 
            display_mode='z', 
            cut_coords=8, 
            colorbar=True,
            black_bg=True,
            symmetric_cbar=symmetric_cbar,
            cmap=cmap
        )
        display.savefig(output_image_path, dpi=dpi, transparent=transparent)
        display.close()
    except Exception as e:
        print(f"Failed to generate plot: {e}")
