import os
import glob
import numpy as np
import nibabel as nib
from nilearn import plotting
import matplotlib.pyplot as plt
from brainiak.isc import isc
from joblib import Parallel, delayed
from scipy.ndimage import label
import config

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
    import time

    print(f"Loading data for condition: {condition}", flush=True)
    data_list = []
    t1 = time.time()

    cond_dir = os.path.join(data_dir, condition)
    mask_xyz = np.where(mask)   # tuple of (x_idx, y_idx, z_idx)
    n_mask_vox = mask_xyz[0].size
    if n_mask_vox == 0:
        raise ValueError("Mask has 0 voxels")

    for sub in subjects:
        search_pattern = os.path.join(cond_dir, f"{sub}_*.nii")

        print(f"[{condition}] {sub}: glob start -> {search_pattern}", flush=True)
        files = sorted(glob.glob(search_pattern))
        print(f"[{condition}] {sub}: glob done ({len(files)} files)", flush=True)

        if not files:
            print(f"Warning: No file found for subject {sub} in {condition}", flush=True)
            continue

        file_path = files[0]

        print(f"[{condition}] {sub}: nib.load start -> {file_path}", flush=True)
        img = nib.load(file_path)

        shape = img.shape          # (X, Y, Z, T)
        x, y, z, t = shape
        print(f"[{condition}] {sub}: image shape {shape}, masked voxels {n_mask_vox}", flush=True)

        print(f"[{condition}] {sub}: load masked voxels start", flush=True)
        t0 = time.time()

        # Read only masked voxels across time
        vox_by_t = np.asanyarray(img.dataobj)[mask_xyz[0], mask_xyz[1], mask_xyz[2], :]   # (n_vox, T)
        masked_data = np.asarray(vox_by_t, dtype=np.float32).T                            # (T, n_vox)
        masked_data = np.nan_to_num(masked_data)

        print(f"[{condition}] {sub}: load masked voxels done in {time.time()-t0:.2f}s shape={masked_data.shape}", flush=True)

        # sanity checks
        if masked_data.shape[0] != t:
            raise ValueError(f"Time dimension mismatch for {sub}: got {masked_data.shape[0]} expected {t}")
        if masked_data.shape[1] != int(mask.sum()):
            raise ValueError(f"Voxel count mismatch for {sub}: got {masked_data.shape[1]} expected {int(mask.sum())}")

        data_list.append(masked_data)

    if not data_list:
        return None

    n_trs = [d.shape[0] for d in data_list]
    min_tr = min(n_trs)

    n_voxels = data_list[0].shape[1]
    n_subs = len(data_list)

    print(f"[{condition}] stacking: n_subs={n_subs}, n_voxels={n_voxels}, min_tr={min_tr}", flush=True)

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

def compute_isc_chunk(chunk_data, pairwise):
    """
    Compute ISC for a single chunk of data.
    """
    # isc() returns shape (n_subjects, n_voxels) for pairwise=False (LOO)
    # isc() returns shape (n_pairs, n_voxels) for pairwise=True
    # Note: brainiak.isc.isc default is pairwise=False (Leave-one-out)
    raw_isc = isc(chunk_data, pairwise=pairwise)
    
    # Compute Fisher-Z for the chunk
    raw_isc_clipped = np.clip(raw_isc, -0.99999, 0.99999)
    z_isc = np.arctanh(raw_isc_clipped)
    
    return raw_isc, z_isc

def run_isc_computation(data, pairwise=False, chunk_size=config.CHUNK_SIZE):
    """
    Run ISC computation in parallel chunks.
    Returns: isc_raw (n_voxels, n_samples), isc_z (n_voxels, n_samples)
             where n_samples is n_subjects (LOO) or n_pairs.
    """
    print(f"Running ISC computation (Pairwise={pairwise}, Chunk Size: {chunk_size})")
    n_trs, n_voxels, n_subs = data.shape
    
    n_chunks = int(np.ceil(n_voxels / chunk_size))
    chunks = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        chunks.append(data[:, start_idx:end_idx, :])

    # Parallel execution
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(compute_isc_chunk)(chunk, pairwise) for chunk in chunks
    )
    
    # Reassemble
    # Determine output shape from first result
    # Result shape from isc is (n_samples, n_chunk_voxels)
    sample_dim = results[0][0].shape[0] # Get sample_dim from raw_isc part of the first result
    
    isc_raw = np.zeros((n_voxels, sample_dim), dtype=np.float32)
    isc_z = np.zeros((n_voxels, sample_dim), dtype=np.float32)
    
    for i, (r_chunk, z_chunk) in enumerate(results):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_voxels)
        
        # Transpose to match (n_voxels, n_samples) for easier slicing later
        isc_raw[start_idx:end_idx, :] = r_chunk.T
        isc_z[start_idx:end_idx, :] = z_chunk.T
        
    return isc_raw, isc_z

def apply_cluster_threshold(img_data, cluster_size):
    """
    Apply cluster-size thresholding to a statistical map.
    img_data: 3D numpy array (thresholded statistical map, where insig voxels are 0)
    cluster_size: Minimum number of voxels for a cluster to be kept.
    """
    if cluster_size <= 0:
        return img_data

    print(f"Applying cluster threshold: k={cluster_size}")
    
    # Binarize for clustering
    binary_map = np.abs(img_data) > 0 
    
    # Label clusters
    # default struct is 3x3x3 connectivity for 3D
    labeled_array, num_features = label(binary_map)
    print(f"  Found {num_features} initial clusters.")
    
    if num_features == 0:
        return img_data
    
    # Calculate sizes
    # labeled_array is 0 for background, 1..N for clusters
    cluster_sizes = np.bincount(labeled_array.ravel())
    
    # Identitfy small clusters (labels) to remove
    # We want to keep labels where size >= cluster_size
    # Note: cluster_sizes[0] is background, we ignore it for "keeping" purposes (it remains 0)
    
    valid_labels = np.where(cluster_sizes >= cluster_size)[0]
    # Filter out 0 (background)
    valid_labels = valid_labels[valid_labels > 0]
    
    # Create a mask of voxels that belong to valid clusters
    valid_mask = np.isin(labeled_array, valid_labels)
    
    cleaned_data = img_data.copy()
    cleaned_data[~valid_mask] = 0
    
    n_kept = len(valid_labels)
    print(f"  Removed {num_features - n_kept} small clusters. Kept {n_kept}.")
    
    return cleaned_data

def apply_tfce(img_data, mask, E=0.5, H=2, dh=0.1, two_sided=True):
    """
    Apply Threshold-Free Cluster Enhancement (TFCE) to a statistical map.
    
    TFCE enhances both signal intensity and spatial extent without requiring
    arbitrary cluster-forming thresholds. It integrates over all possible
    thresholds to create an enhanced statistic map.
    
    Parameters:
    -----------
    img_data : 3D numpy array
        Statistical map (e.g., t-values, z-scores). Should be 3D (X, Y, Z).
    mask : 3D boolean numpy array
        Brain mask indicating valid voxels. Same shape as img_data.
    E : float, optional (default=0.5)
        Extent parameter. Controls weight given to cluster size.
    H : float, optional (default=2)
        Height parameter. Controls weight given to statistical height.
    dh : float, optional (default=0.1)
        Step size for threshold integration. Smaller = more accurate but slower.
    two_sided : bool, optional (default=True)
        If True, process positive and negative values separately and combine.
        If False, only process positive values.
    
    Returns:
    --------
    tfce_map : 3D numpy array
        TFCE-enhanced statistical map. Same shape as img_data.
    
    References:
    -----------
    Smith & Nichols (2009). Threshold-free cluster enhancement: 
    addressing problems of smoothing, threshold dependence and 
    localisation in cluster inference. NeuroImage, 44(1), 83-98.
    """
    print(f"Applying TFCE (E={E}, H={H}, dh={dh}, two_sided={two_sided})...")
    
    # Initialize output
    tfce_map = np.zeros_like(img_data, dtype=np.float32)
    
    # Mask the data
    masked_data = img_data.copy()
    masked_data[~mask] = 0
    
    if two_sided:
        # Process positive and negative values separately
        # Positive values
        pos_data = np.maximum(masked_data, 0)
        if np.any(pos_data > 0):
            tfce_pos = _compute_tfce_single_direction(pos_data, mask, E, H, dh, direction='positive')
            tfce_map += tfce_pos
        
        # Negative values (take absolute value, then negate result)
        neg_data = np.maximum(-masked_data, 0)
        if np.any(neg_data > 0):
            tfce_neg = _compute_tfce_single_direction(neg_data, mask, E, H, dh, direction='positive')
            tfce_map -= tfce_neg  # Negate because these were negative values
    else:
        # Only process positive values
        pos_data = np.maximum(masked_data, 0)
        if np.any(pos_data > 0):
            tfce_map = _compute_tfce_single_direction(pos_data, mask, E, H, dh, direction='positive')
    
    print(f"  TFCE complete. Range: [{np.nanmin(tfce_map[mask]):.3f}, {np.nanmax(tfce_map[mask]):.3f}]")
    
    return tfce_map

def _compute_tfce_single_direction(data, mask, E, H, dh, direction='positive'):
    """
    Compute TFCE for a single direction (positive values only).
    
    This is a helper function that does the actual TFCE computation.
    """
    tfce_map = np.zeros_like(data, dtype=np.float32)
    
    # Get valid (non-zero) values within mask
    valid_voxels = (data > 0) & mask
    if not np.any(valid_voxels):
        return tfce_map
    
    # Get range of values to threshold over
    min_val = np.min(data[valid_voxels])
    max_val = np.max(data[valid_voxels])
    
    if min_val >= max_val:
        return tfce_map
    
    # Create threshold levels
    # Start slightly above min to avoid numerical issues
    thresholds = np.arange(min_val + dh, max_val + dh, dh)
    
    if len(thresholds) == 0:
        return tfce_map
    
    # Process each threshold level
    for h in thresholds:
        # Threshold: keep voxels above current threshold
        thresholded = data >= h
        thresholded = thresholded & mask  # Ensure within mask
        
        if not np.any(thresholded):
            continue
        
        # Find connected clusters at this threshold
        labeled_array, num_clusters = label(thresholded)
        
        if num_clusters == 0:
            continue
        
        # Calculate cluster sizes
        cluster_sizes = np.bincount(labeled_array.ravel())
        
        # For each cluster, add contribution to TFCE map
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = (labeled_array == cluster_id)
            cluster_size = cluster_sizes[cluster_id]
            
            # TFCE contribution: (cluster_size^E) * (threshold^H) * dh
            contribution = (cluster_size ** E) * (h ** H) * dh
            
            # Add contribution to all voxels in this cluster
            tfce_map[cluster_mask] += contribution
    
    return tfce_map

def apply_fdr(p_values, alpha=0.05):
    """
    Apply FDR correction (Benjamini-Hochberg) to p-values.
    """
    p_flat = np.asarray(p_values).flatten()
    m = len(p_flat)
    
    # Handle NaNs: mask them out
    mask = ~np.isnan(p_flat)
    p_valid = p_flat[mask]
    m_valid = len(p_valid)
    
    if m_valid == 0:
        return np.nan * np.ones_like(p_values), np.zeros_like(p_values, dtype=bool)
    
    # Sort p-values
    sort_inds = np.argsort(p_valid)
    p_sorted = p_valid[sort_inds]
    
    # Calculate q-values using BH procedure
    ranks = np.arange(1, m_valid + 1)
    q_valid = p_sorted * m_valid / ranks
    
    # Ensure monotonicity: q_i = min(q_i, q_{i+1}) backwards
    q_valid = np.minimum.accumulate(q_valid[::-1])[::-1]
    q_valid = np.minimum(q_valid, 1.0)
    
    # Map back to original order
    q_flat = np.nan * np.ones(m)
    reject_flat = np.zeros(m, dtype=bool)
    
    q_valid_unsorted = np.zeros(m_valid)
    q_valid_unsorted[sort_inds] = q_valid
    
    q_flat[mask] = q_valid_unsorted
    reject_flat[mask] = q_flat[mask] < alpha
    
    return q_flat.reshape(np.asarray(p_values).shape), reject_flat.reshape(np.asarray(p_values).shape)

def apply_bonferroni(p_values, alpha=0.05):
    """
    Apply Bonferroni correction to p-values.
    """
    p_flat = np.asarray(p_values).flatten()
    mask = ~np.isnan(p_flat)
    n_tests = np.sum(mask)
    
    p_corrected = np.nan * np.ones_like(p_flat)
    p_corrected[mask] = p_flat[mask] * n_tests
    p_corrected = np.minimum(p_corrected, 1.0)
    
    reject = p_corrected < alpha
    return p_corrected.reshape(np.asarray(p_values).shape), reject.reshape(np.asarray(p_values).shape)

