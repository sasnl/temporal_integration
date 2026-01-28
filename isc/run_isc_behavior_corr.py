
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import sys
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
import time

# Import config and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
import shared.config as config
from shared.pipeline_utils import load_mask, save_map, save_plot, apply_cluster_threshold, apply_tfce

def parse_args():
    parser = argparse.ArgumentParser(description='Run Correlation between ISC/ISFC Maps and Behavioral Scores')
    
    # Input Data
    parser.add_argument('--condition', type=str, required=True, help='Condition Name (e.g., TI1_orig). If contrast is used, this is Condition 1.')
    parser.add_argument('--contrast', type=str, default=None, help='Optional: Condition 2 for Contrast (Cond1 - Cond2)')
    
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with behavioral data')
    parser.add_argument('--behavior_col', type=str, required=True, help='Column name for behavioral score')
    
    parser.add_argument('--isc_method', type=str, choices=['loo', 'pairwise'], default='loo', help='ISC Method (default: loo)')
    parser.add_argument('--roi_id', type=int, default=None, help='ROI ID for masking')
    parser.add_argument('--data_dir', type=str, default=config.OUTPUT_DIR, help='Directory containing ISC results')
    
    # Stats
    parser.add_argument('--corr_method', type=str, choices=['pearson', 'spearman'], default='pearson', help='Correlation Type')
    parser.add_argument('--n_perms', type=int, default=1000, help='Number of permutations')
    
    # Correction
    parser.add_argument('--p_threshold', type=float, default=0.05, help='P-value threshold')
    parser.add_argument('--cluster_threshold', type=int, default=0, help='Cluster threshold (voxels)')
    parser.add_argument('--use_tfce', action='store_true', help='Use TFCE correction')
    parser.add_argument('--tfce_E', type=float, default=0.5)
    parser.add_argument('--tfce_H', type=float, default=2.0)
    
    # Output
    parser.add_argument('--output_dir', type=str, default=os.path.join(config.OUTPUT_DIR, 'ISC_behav'), help='Output directory')
    
    return parser.parse_args()

def load_subjects_and_behavior(csv_path, col_name, condition, contrast_cond=None):
    """
    Loads behavioral data and intersects with subject lists.
    Returns: 
        common_subs (list), 
        behavior_scores (array), 
        list1 (full subject list for cond1), 
        list2 (full subject list for cond2 or None)
    """
    print(f"Loading behavioral data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check column
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in CSV. Available: {list(df.columns)}")
    
    # Ensure record_id is string for matching
    if 'record_id' not in df.columns:
         raise ValueError("CSV must contain 'record_id' column")
    df['record_id'] = df['record_id'].astype(str)
    
    # Load Config Subjects
    if condition not in config.SUBJECT_LISTS:
        raise ValueError(f"Condition {condition} not found in config.SUBJECT_LISTS")
    list1 = [str(s) for s in config.SUBJECT_LISTS[condition]]
    
    valid_subs_set = set(list1)
    
    if contrast_cond:
        if contrast_cond not in config.SUBJECT_LISTS:
             raise ValueError(f"Condition {contrast_cond} not found in config.SUBJECT_LISTS")
        list2 = [str(s) for s in config.SUBJECT_LISTS[contrast_cond]]
        valid_subs_set = valid_subs_set.intersection(set(list2))
    else:
        list2 = None
        
    # Intersect with CSV
    # Filter DF to only include valid subjects and non-NaN scores
    df_filtered = df[df['record_id'].isin(valid_subs_set) & df[col_name].notna()]
    
    common_subs = sorted(df_filtered['record_id'].unique().tolist())
    
    # Align scores
    # Create a mapping dict
    score_map = dict(zip(df_filtered['record_id'], df_filtered[col_name]))
    behavior_scores = np.array([score_map[s] for s in common_subs])
    
    print(f"Subjects Alignment:")
    print(f"  Condition 1 ({condition}): {len(list1)}")
    if contrast_cond:
        print(f"  Condition 2 ({contrast_cond}): {len(list2)}")
    print(f"  Final Analyzed Subjects: {len(common_subs)}")
    
    if len(common_subs) < 3:
        raise ValueError("Not enough subjects for correlation (<3)!")
        
    return common_subs, behavior_scores, list1, list2

def load_brain_data(condition, isc_method, roi_id, data_dir, full_subject_list, common_subs, mask_file):
    """
    Loads 4D ISC map and extracts subjects.
    """
    roi_suffix = f"_roi{roi_id}" if roi_id is not None else ""
    # Search for file in known subdirs if not provided directly
    # The user might pass 'result/ISC' but files are deep.
    # We'll try a few standard locations if direct path fails.
    
    fname = f"isc_{condition}_{isc_method}{roi_suffix}_desc-zscore.nii.gz"
    
    # Try multiple possible paths based on known structure
    candidates = [
        os.path.join(data_dir, fname),
        os.path.join(data_dir, 'ISC_bootstrap', 'fdr', isc_method, fname), # HACK: Just finding one that exists
        os.path.join(data_dir, 'ISC_bootstrap', 'fwe', isc_method, fname),
        os.path.join(data_dir, 'ISC_bootstrap', 'tfce', isc_method, fname) 
    ]
    
    fpath = None
    for p in candidates:
        if os.path.exists(p):
            fpath = p
            break
            
    if fpath is None:
        # Fallback: exact find
        import glob
        search = os.path.join(data_dir, "**", fname)
        found = glob.glob(search, recursive=True)
        if found:
            fpath = found[0]
        else:
            raise FileNotFoundError(f"Could not find ISC map {fname} in {data_dir}")
            
    print(f"Loading Map: {fpath}")
    img = nib.load(fpath)
    data_full = img.get_fdata(dtype=np.float32)
    affine = img.affine
    
    mask_data, _ = load_mask(mask_file, roi_id)
    
    # Check Dims
    # If 4D (X,Y,Z,S)
    if len(data_full.shape) == 4:
        # Valid
        pass
    else:
         raise ValueError(f"Expected 4D map, got {data_full.shape}")
         
    # Flatten via Mask
    # data_full[mask]: (V, S_total)
    # Check subject count
    if data_full.shape[3] != len(full_subject_list):
        print(f"Warning: Map has {data_full.shape[3]} volumes, but config lists {len(full_subject_list)} subjects.")
        # We assume they match order. If count differs, it's risky.
        if data_full.shape[3] < len(full_subject_list):
             raise ValueError("Map has fewer subjects than config list! Cannot align safely.")
    
    # Extract
    indices = [full_subject_list.index(s) for s in common_subs]
    data_flat = data_full[mask_data][:, indices] # (V, N_common)
    
    return data_flat, mask_data, affine

def compute_correlation(brain_data, behavior, method='pearson'):
    """
    Computes correlation between (V, S) brain data and (S,) behavior.
    """
    n_voxels, n_subs = brain_data.shape
    
    # Center data for efficiency? r = dot(x_c, y_c) / (norm_x * norm_y)
    # But scipy is fast enough or use numpy for vectorization
    
    # Vectorized Pearson
    if method == 'pearson':
        # Center rows
        X = brain_data - np.nanmean(brain_data, axis=1, keepdims=True)
        Y = behavior - np.nanmean(behavior) # (S,)
        
        # Norms
        # X: (V, S)
        norm_x = np.sqrt(np.nansum(X**2, axis=1))
        norm_y = np.sqrt(np.nansum(Y**2))
        
        numerator = np.nansum(X * Y, axis=1) # (V,)
        r = numerator / (norm_x * norm_y)
        
    elif method == 'spearman':
        # Scipy spearman is slower, loop or apply_along_axis
        # Or rank data first then Pearson
        from scipy.stats import rankdata
        
        # Rank behavior
        y_ranked = rankdata(behavior)
        
        # Rank brain (row-wise)
        # Apply rankdata along axis 1
        # Handle NaNs in ranking? rankdata supports nan_policy?
        # For simplicity, if NaNs exist, rankdata might propagate or rank them.
        # We'll use a nan-safe apply.
        def _rank_safe(row):
            if np.isnan(row).any():
                # Mask NaNs
                valid = ~np.isnan(row)
                r = np.empty_like(row)
                r[:] = np.nan
                r[valid] = rankdata(row[valid])
                return r
            else:
                return rankdata(row)
                
        X_ranked = np.apply_along_axis(_rank_safe, 1, brain_data)
        
        # Compute Pearson on ranks
        return compute_correlation(X_ranked, y_ranked, 'pearson')
        
    return r

def run_permutation(brain_data, behavior, n_perms, corr_method, mask_data, use_tfce, tfce_E, tfce_H):
    """
    Permutes behavior vector.
    """
    print(f"Running Permutation Test (n={n_perms})...")
    
    n_voxels, n_subs = brain_data.shape
    rng = np.random.RandomState(42)
    
    null_max_stats = []
    null_voxels = [] # Only if not TFCE and we want uncorrected p-values? 
                     # Actually for uncorrected, we can just use parametric p-value or voxelwise null.
    
    # Optimization: Pre-calculate brain ranks if Spearman
    X_for_corr = brain_data
    if corr_method == 'spearman':
         from scipy.stats import rankdata
         X_for_corr = np.apply_along_axis(rankdata, 1, brain_data)
        
    # Pre-center X for Pearson
    X_c = X_for_corr - np.nanmean(X_for_corr, axis=1, keepdims=True)
    norm_x = np.sqrt(np.nansum(X_c**2, axis=1))
    
    # Compute Observed R
    r_obs = compute_correlation(brain_data, behavior, corr_method)
    
    # Convert r to t-stat for TFCE (better dynamic range)
    # t = r * sqrt(n-2) / sqrt(1-r^2)
    # Avoid div/0
    r_obs_safe = np.clip(r_obs, -0.999999, 0.999999)
    df = n_subs - 2
    t_obs = r_obs_safe * np.sqrt(df) / np.sqrt(1 - r_obs_safe**2)
    
    stat_obs = t_obs if use_tfce else r_obs
    
    # TFCE Observed
    if use_tfce:
        stat_obs_3d = np.zeros(mask_data.shape, dtype=np.float32)
        stat_obs_3d[mask_data] = stat_obs
        stat_obs_3d = apply_tfce(stat_obs_3d, mask_data, E=tfce_E, H=tfce_H, two_sided=True)
        metric_obs = np.max(np.abs(stat_obs_3d)) # Just for tracking? No, this is the map.
        perm_input_map = stat_obs_3d
    else:
        perm_input_map = stat_obs
    
    # Parallel Permutations
    def _perm_loop(i):
        # Shuffle behavior
        y_perm = rng.permutation(behavior)
        
        if corr_method == 'spearman':
            from scipy.stats import rankdata
            y_perm = rankdata(y_perm)
            
        # Pearson logic
        y_c = y_perm - np.mean(y_perm)
        norm_y = np.sqrt(np.sum(y_c**2))
        
        r_perm = np.dot(X_c, y_c) / (norm_x * norm_y)
        
        if use_tfce:
            r_perm = np.clip(r_perm, -0.999999, 0.999999)
            t_perm = r_perm * np.sqrt(df) / np.sqrt(1 - r_perm**2)
            
            t_perm_3d = np.zeros(mask_data.shape, dtype=np.float32)
            t_perm_3d[mask_data] = t_perm
            # TFCE
            t_perm_3d = apply_tfce(t_perm_3d, mask_data, E=tfce_E, H=tfce_H, two_sided=True)
            return np.max(np.abs(t_perm_3d))
        else:
            # Just return max |r| for FWER? Or return whole map for voxelwise?
            # To save memory, let's just return max for FWER, 
            # and rely on parametric p-values for "uncorrected" map (since n=21 is decent for parametric).
            return np.max(np.abs(r_perm))

    null_max = Parallel(n_jobs=-1, verbose=5)(
        delayed(_perm_loop)(i) for i in range(n_perms)
    )
    null_max = np.array(null_max)
    
    # Calculate Parametric Uncorrected P-values regardless of Permutation method
    # t = r * sqrt(n-2) / sqrt(1-r^2)
    from scipy.stats import t as t_dist
    p_unc = 2 * (1 - t_dist.cdf(np.abs(t_obs), df))

    # Calculate P-values
    # 1. TFCE FWER P
    if use_tfce:
        # perm_input_map is the TFCE-enhanced Observed Map (3D)
        # Compare voxels against Null Max Distribution
        obs_flat = perm_input_map[mask_data] # Back to 1D
        
        p_fwer = (np.sum(null_max[:, np.newaxis] >= np.abs(obs_flat[np.newaxis, :]), axis=0) + 1) / (n_perms + 1)
        
        return r_obs, p_unc, p_fwer # Return Raw R, P_Unc, P_FWER
    else:
        return r_obs, p_unc, None


def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # 1. Subjects & Behavior
    common_subs, behavior, list1, list2 = load_subjects_and_behavior(
        args.csv_file, args.behavior_col, args.condition, args.contrast
    )
    
    # 2. Load Brain Data
    # Cond 1
    d1, mask_data, affine = load_brain_data(
        args.condition, args.isc_method, args.roi_id, args.data_dir, list1, common_subs, config.MASK_FILE
    )
    
    final_data = d1
    
    if args.contrast:
        # Cond 2
        d2, _, _ = load_brain_data(
            args.contrast, args.isc_method, args.roi_id, args.data_dir, list2, common_subs, config.MASK_FILE
        )
        # Compute Contrast
        final_data = d1 - d2
        print(f"Computing Correlation on Contrast: {args.condition} - {args.contrast}")
        
    else:
        print(f"Computing Correlation on Condition: {args.condition}")
        
    # 3. Correlation & Stats
    r_map, p_unc, p_fwer = run_permutation(
        final_data, behavior, args.n_perms, args.corr_method, 
        mask_data, args.use_tfce, args.tfce_E, args.tfce_H
    )
    
    # 4. Save
    base_name = f"corr_{args.condition}"
    if args.contrast:
        base_name += f"_vs_{args.contrast}"
    base_name += f"_{args.behavior_col}_{args.corr_method}"
    
    if args.use_tfce:
        base_name += "_tfce"
        
    # Save R map
    r_path = os.path.join(args.output_dir, f"{base_name}_desc-r.nii.gz")
    save_map(r_map, mask_data, affine, r_path)
    
    # Save Uncorrected P map
    p_unc_path = os.path.join(args.output_dir, f"{base_name}_desc-pvalues_uncorrected.nii.gz")
    p_unc_3d = np.ones(mask_data.shape, dtype=np.float32)
    p_unc_3d[mask_data] = p_unc
    save_map(p_unc_3d, mask_data, affine, p_unc_path)
    
    # Decide which P-values to use for significance map
    if p_fwer is not None:
        p_final = p_fwer
        # Save FWER P map
        p_fwer_path = os.path.join(args.output_dir, f"{base_name}_desc-pvalues.nii.gz")
        p_fwer_3d = np.ones(mask_data.shape, dtype=np.float32)
        p_fwer_3d[mask_data] = p_fwer
        save_map(p_fwer_3d, mask_data, affine, p_fwer_path)
        print(f"Saving FWER Corrected P-values to {p_fwer_path}")
    else:
        p_final = p_unc
        print(f"Saving Uncorrected P-values to {p_unc_path}")
    
    # Save Significant R Map (Corrected if TFCE, else Uncorrected)
    sig_r = r_map.copy()
    sig_r[p_final >= args.p_threshold] = 0
    
    # Save Uncorrected Significant Map as well (if using TFCE)
    if args.use_tfce:
        sig_r_unc = r_map.copy()
        sig_r_unc[p_unc >= args.p_threshold] = 0
        sig_unc_path = os.path.join(args.output_dir, f"{base_name}_desc-sig_uncorrected_p{str(args.p_threshold).replace('.', '')}.nii.gz")
        
        # We don't cluster threshold uncorrected map here by default, or should we?
        # User asked to "plot the significant correlation map uncorrected".
        # Let's clean up NaNs before saving
        sig_r_unc = np.nan_to_num(sig_r_unc)
        
        # Save 3D
        sig_r_unc_3d = np.zeros(mask_data.shape, dtype=np.float32)
        sig_r_unc_3d[mask_data] = sig_r_unc
        save_map(sig_r_unc_3d, mask_data, affine, sig_unc_path)
        
        # Plot Uncorrected
        plot_unc_path = os.path.join(args.output_dir, f"{base_name}_desc-sig_uncorrected.png")
        save_plot(sig_unc_path, plot_unc_path, f"Correlation (Uncorrected p<{args.p_threshold}): {args.condition} vs {args.behavior_col}", positive_only=False)
        print(f"Saved Uncorrected Significant Plot to {plot_unc_path}")

    # Cluster Correction (Optional)
    if args.cluster_threshold > 0:
        print(f"Applying Cluster Threshold: {args.cluster_threshold}")
        # Need 3D R map
        r_3d = np.zeros(mask_data.shape, dtype=np.float32)
        r_3d[mask_data] = sig_r
        
        # Apply cluster
        r_3d_clus = apply_cluster_threshold(r_3d, args.cluster_threshold)
        sig_r_to_save = r_3d_clus
    else:
        sig_r_to_save = np.zeros(mask_data.shape, dtype=np.float32)
        sig_r_to_save[mask_data] = sig_r
        
    sig_path = os.path.join(args.output_dir, f"{base_name}_desc-sig_p{str(args.p_threshold).replace('.', '')}.nii.gz")
    save_map(sig_r_to_save, mask_data, affine, sig_path)
    
    # Plot
    plot_path = os.path.join(args.output_dir, f"{base_name}_desc-sig.png")
    title = f"Correlation: {args.condition} vs {args.behavior_col} ({args.corr_method})"
    save_plot(sig_path, plot_path, title, positive_only=False)
    
    # NaN Check
    nan_count = np.isnan(r_map).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} voxels ({nan_count/r_map.size:.2%}) have NaN correlation values.")
    
    print("Done.")
    print(f"Outputs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
