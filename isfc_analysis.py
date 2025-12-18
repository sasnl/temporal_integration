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
# Set ROI_ID to a specific integer to run ISFC only on that region
# If None, runs on valid voxels in mask (WARNING: Whole brain ISFC is huge!)
ROI_ID = None  # Defaulting to None unless specific ROI analysis is needed

# Seed-based ISFC Configuration
# If SEED_COORD is set, the script will run Seed-to-Voxel ISFC (generating a brain map).
# ROI_ID will be ignored if SEED_COORD is set.
# Coordinates should be in the same space as the mask (e.g., MNI mm).
SEED_COORD = (-2, -54, 26)  # Example: PCC
SEED_RADIUS = 3  # Radius in mm. Set to 0 for single voxel.

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

def coord_to_voxel(affine, coords):
    """
    Convert world coordinates to voxel indices.
    """
    # Inverse affine gives voxel coordinates
    inv_affine = np.linalg.inv(affine)
    # Add homogeneous coordinate (1)
    coords_h = np.array(list(coords) + [1])
    voxel_coords = inv_affine @ coords_h
    # Round to nearest integer
    return np.round(voxel_coords[:3]).astype(int)

def get_seed_mask(mask_shape, affine, center_coords, radius_mm):
    """
    Create a binary mask for a spherical seed.
    """
    print(f"Creating spherical seed mask at {center_coords} with r={radius_mm}mm")
    
    # Create a grid of coordinates
    nx, ny, nz = mask_shape
    # We need to compute the world distance of every voxel from the center_coords.
    # This can be expensive for a full brain, so we can optimize by only checking a bounding box
    # or just brute forcing it if the volume isn't massive.
    # Vectorized approach:
    
    # 1. Get voxel indices of the center
    center_vox = coord_to_voxel(affine, center_coords)
    print(f"  Center voxel indices: {center_vox}")
    
    # 2. To avoid iterating all voxels, we can define a bounding box in voxel space
    # Approximation: divide radius by min voxel size to get voxel radius
    vox_sizes = np.sqrt(np.sum(affine[:3, :3]**2, axis=0))
    min_vox_size = np.min(vox_sizes)
    voxel_radius = int(np.ceil(radius_mm / min_vox_size)) + 1
    
    min_x = max(0, center_vox[0] - voxel_radius)
    max_x = min(nx, center_vox[0] + voxel_radius + 1)
    min_y = max(0, center_vox[1] - voxel_radius)
    max_y = min(ny, center_vox[1] + voxel_radius + 1)
    min_z = max(0, center_vox[2] - voxel_radius)
    max_z = min(nz, center_vox[2] + voxel_radius + 1)
    
    seed_mask = np.zeros(mask_shape, dtype=bool)
    
    # Check distances within bounding box
    count = 0
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            for z in range(min_z, max_z):
                # Convert this voxel back to world coordinates
                # Or simply compute distance? Distance in mm is requested.
                # It's safer to convert voxel -> world to compute Euclidean distance in mm.
                pt_vox = np.array([x, y, z, 1])
                pt_world = affine @ pt_vox
                dist = np.linalg.norm(pt_world[:3] - np.array(center_coords))
                
                if dist <= radius_mm:
                    seed_mask[x, y, z] = True
                    count += 1
                    
    print(f"  Seed mask contains {count} voxels")
    if count == 0:
        # Fallback to just the center voxel if radius is too small or something failed
        print("  WARNING: Sphere empty, falling back to single voxel.")
        if 0 <= center_vox[0] < nx and 0 <= center_vox[1] < ny and 0 <= center_vox[2] < nz:
            seed_mask[center_vox[0], center_vox[1], center_vox[2]] = True
        else:
            raise ValueError("Seed coordinate is out of brain bounds.")
            
    return seed_mask

def load_seed_data(group_data_full_brain, seed_mask, whole_brain_mask):
    """
    Extract seed time series from the already loaded group data.
    group_data_full_brain: (TRs, Voxels, Subjects) corresponding to whole_brain_mask
    seed_mask: Full 3D boolean mask of the seed
    whole_brain_mask: Full 3D boolean mask of the whole analysis space
    """
    print("Extracting seed time series...")
    
    # Vectorize the seed mask exactly like the whole brain mask was vectorized
    # We need to know which indices in 'group_data_full_brain' correspond to 'seed_mask'
    
    # The whole_brain_mask was used to flatten the data: flattened = data[whole_brain_mask]
    # So we need to find the subset of indices in flattened that correspond to seed_mask=True
    
    # 1. Get all voxel indices in 3D
    # It's easier to just reload/mask raw data, BUT we already loaded 'group_data'.
    # However, 'group_data' only contains voxels where whole_brain_mask is True.
    # So we need to ensure the seed is within the whole_brain_mask.
    
    valid_seed_mask = seed_mask & whole_brain_mask
    if np.sum(valid_seed_mask) == 0:
        raise ValueError("Seed mask does not overlap with the analysis brain mask!")

    # Where in the "valid voxels list" are the seed voxels?
    # We can use np.where to map 3D indices to 1D flattened indices.
    
    # Full mapping 3D -> 1D index
    # -1 indicates voxel is not in the mask
    vol_shape = whole_brain_mask.shape
    mapping_3d_to_1d = np.full(vol_shape, -1, dtype=int)
    mapping_3d_to_1d[whole_brain_mask] = np.arange(np.sum(whole_brain_mask))
    
    # Get 1D indices for the seed
    seed_indices_1d = mapping_3d_to_1d[valid_seed_mask]
    
    print(f"  Seed comprises {len(seed_indices_1d)} valid voxels from the Analysis ROI.")
    
    # Extract data: (TRs, SeedVoxels, Subjects)
    seed_voxels_data = group_data_full_brain[:, seed_indices_1d, :]
    
    # Average across seed voxels -> (TRs, 1, Subjects)
    seed_timeseries = np.mean(seed_voxels_data, axis=1, keepdims=True)
    
    return seed_timeseries

def run_isfc(data, targets=None):
    """
    Compute ISFC (Inter-Subject Functional Correlation).
    
    Parameters
    ----------
    data : array_like (TRs, Voxels_1, Subjects)
        The source data (e.g., Seed).
    targets : array_like (TRs, Voxels_2, Subjects), optional
        The target data (e.g., Whole Brain). If None, computes auto-ISFC on `data`.
    
    Returns
    -------
    isfc_matrix : ndarray 
        If targets is None: (Voxels_1, Voxels_1)
        If targets is provided: (Voxels_1, Voxels_2)
        The group-average leave-one-out ISFC matrix.
    """
    print("Running ISFC analysis...")
    # brainiak.isc.isfc computes the ISFC.
    # pairwise=False: computes leave-one-out ISFCs (one map per subject).
    # summary_statistic='mean': averages these maps into a single group matrix.
    
    if targets is None:
        isfc_matrix = isfc(data, pairwise=False, summary_statistic='mean', vectorize_isfcs=False)
    else:
        print(f"  Computing Seed-to-Voxel ISFC (Source: {data.shape[1]} voxels, Target: {targets.shape[1]} voxels)")
        isfc_matrix = isfc(data, targets, pairwise=False, summary_statistic='mean', vectorize_isfcs=False)
    
    return isfc_matrix

def save_matrix(matrix, output_path):
    print(f"Saving ISFC matrix to {output_path}")
    print(f"  Matrix shape: {matrix.shape}")
    np.save(output_path, matrix)

def save_nifti(vector_data, mask_file, output_path):
    """
    Save specific results as NIfTI images using the mask as a template.
    vector_data: 1D array of length equal to np.sum(load_mask(mask_file))
    """
    print(f"Saving NIfTI map to {output_path}")
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata().astype(bool)
    
    # Initialize empty volume
    out_data = np.zeros(mask_data.shape, dtype=np.float32)
    
    # Fill mask
    # If vector_data is shape (1, N) or (N, 1), flatten it
    out_data[mask_data] = vector_data.flatten()
    
    out_img = nib.Nifti1Image(out_data, mask_img.affine, mask_img.header)
    nib.save(out_img, output_path)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Check mode
    if 'SEED_COORD' in globals() and SEED_COORD is not None:
        mode = "seed"
        print(f"--- MODE: SEED-BASED ISFC (Seed: {SEED_COORD}, Radius: {SEED_RADIUS if 'SEED_RADIUS' in globals() else 0}mm) ---")
        # For seed mode, we generally want the whole brain as the target
        mask_roi_id = None 
    else:
        mode = "matrix"
        print(f"--- MODE: ROI-ROI MATRIX ISFC (ROI_ID: {ROI_ID}) ---")
        mask_roi_id = ROI_ID

    mask, affine = load_mask(MASK_FILE, roi_id=mask_roi_id)
    
    if np.sum(mask) == 0:
        print("Error: Mask is empty. Exiting.")
        return
        
    n_voxels = np.sum(mask)
    
    if mode == "matrix":
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
        
        if mode == "seed":
            # 1. Generate seed mask
            radius = SEED_RADIUS if 'SEED_RADIUS' in globals() else 0
            seed_mask = get_seed_mask(mask.shape, affine, SEED_COORD, radius)
            
            # 2. Extract seed timeseries
            seed_ts = load_seed_data(group_data, seed_mask, mask)
            # Shape (TRs, 1, Subjects)
            
            # 3. Run ISFC (Seed vs Whole Brain)
            # group_data is (TRs, Voxels, Subjects)
            isfc_result = run_isfc(seed_ts, targets=group_data)
            # Result shape: (1, Voxels)
            
            # 4. Save as NIfTI
            output_filename = f"isfc_seed{SEED_COORD[0]}_{SEED_COORD[1]}_{SEED_COORD[2]}_r{radius}_{condition}.nii.gz"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            save_nifti(isfc_result, MASK_FILE, output_path)
            
        else:
            # Matrix mode
            isfc_result = run_isfc(group_data)
            
            # Modify output filename
            suffix = f"_roi{ROI_ID}" if ROI_ID is not None else "_wholebrain"
            output_filename = f"isfc_{condition}{suffix}.npy"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
        
            save_matrix(isfc_result, output_path)
        
        print(f"Finished {condition} in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
