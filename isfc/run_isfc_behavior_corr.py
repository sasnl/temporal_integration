
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import sys
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed
import time
import re

# Import config and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
import shared.config as config
from shared.pipeline_utils import load_mask, save_map, save_plot, apply_cluster_threshold, apply_tfce

def parse_args():
    parser = argparse.ArgumentParser(description='Run Correlation between ISFC Maps and Behavioral Scores')
    
    # Input Data
    parser.add_argument('--input_file', type=str, required=True, help='Full path to 4D ISFC Z-score map (Condition 1)')
    parser.add_argument('--contrast_file', type=str, default=None, help='Optional: Full path to 4D ISFC Z-score map for Condition 2 (Cond1 - Cond2)')
    
    parser.add_argument('--condition', type=str, required=True, help='Condition Name (e.g., TI1_orig). If contrast is used, this is Condition 1.')
    parser.add_argument('--contrast_condition', type=str, default=None, help='Optional: Condition 2 Name for Contrast')
    
    parser.add_argument('--csv_file', type=str, required=True, help='Path to CSV file with behavioral data')
    parser.add_argument('--behavior_col', type=str, required=True, help='Column name for behavioral score')
    
    parser.add_argument('--roi_id', type=int, default=None, help='ROI ID for masking (if input map is full brain, optional)')
    
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
    parser.add_argument('--output_dir', type=str, default=os.path.join(config.OUTPUT_DIR, 'ISFC_behav'), help='Output directory')
    
    return parser.parse_args()

def load_subjects_and_behavior(csv_path, col_name, condition, contrast_cond=None):
    """
    Loads behavioral data and intersects with subject lists.
    """
    print(f"Loading behavioral data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if col_name not in df.columns:
        raise ValueError(f"Column '{col_name}' not found in CSV.")
    
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
    df_filtered = df[df['record_id'].isin(valid_subs_set) & df[col_name].notna()]
    
    common_subs = sorted(df_filtered['record_id'].unique().tolist())
    
    # Align scores
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

def load_brain_data(file_path, full_subject_list, common_subs, mask_file, roi_id=None):
    """
    Loads 4D ISFC map from specific file and extracts subjects.
    """
    print(f"Loading Map: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    img = nib.load(file_path)
    data_full = img.get_fdata(dtype=np.float32)
    affine = img.affine
    
    # Load mask
    mask_data, _ = load_mask(mask_file, roi_id)
    
    # Check Dims
    if len(data_full.shape) != 4:
         raise ValueError(f"Expected 4D map, got {data_full.shape}")
         
    # Check subject count
    if data_full.shape[3] != len(full_subject_list):
        print(f"Warning: Map has {data_full.shape[3]} volumes, but config lists {len(full_subject_list)} subjects.")
        if data_full.shape[3] < len(full_subject_list):
             raise ValueError("Map has fewer subjects than config list! Cannot align safely.")
    
    # Extract subjects
    indices = [full_subject_list.index(s) for s in common_subs]
    
    # If mask matches 3D dims
    if data_full.shape[:3] != mask_data.shape:
        raise ValueError(f"Map dims {data_full.shape[:3]} do not match mask {mask_data.shape}")

    data_flat = data_full[mask_data][:, indices] # (V, N_common)
    
    return data_flat, mask_data, affine

def compute_correlation(brain_data, behavior, method='pearson'):
    """
    Computes correlation between (V, S) brain data and (S,) behavior.
    """
    n_voxels, n_subs = brain_data.shape
    
    if method == 'pearson':
        # Vectorized Pearson with NaN safety
        X = brain_data - np.nanmean(brain_data, axis=1, keepdims=True)
        Y = behavior - np.nanmean(behavior) # (S,)
        
        # Norms
        norm_x = np.sqrt(np.nansum(X**2, axis=1))
        norm_y = np.sqrt(np.nansum(Y**2))
        
        numerator = np.nansum(X * Y, axis=1) # (V,)
        r = numerator / (norm_x * norm_y)
        
    elif method == 'spearman':
        from scipy.stats import rankdata
        y_ranked = rankdata(behavior)
        
        def _rank_safe(row):
            if np.isnan(row).any():
                valid = ~np.isnan(row)
                r = np.empty_like(row)
                r[:] = np.nan
                r[valid] = rankdata(row[valid])
                return r
            else:
                return rankdata(row)
                
        X_ranked = np.apply_along_axis(_rank_safe, 1, brain_data)
        return compute_correlation(X_ranked, y_ranked, 'pearson')
        
    return r

def run_permutation(brain_data, behavior, n_perms, corr_method, mask_data, use_tfce, tfce_E, tfce_H):
    """
    Permutes behavior vector. NaN-safe. Fisher-Z TFCE.
    """
    print(f"Running Permutation Test (n={n_perms})...")
    
    n_voxels, n_subs = brain_data.shape
    
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
    
    # Convert r to Fisher Z for TFCE
    r_obs_safe = np.clip(r_obs, -0.999999, 0.999999)
    z_obs = np.arctanh(r_obs_safe)
    
    stat_obs = z_obs if use_tfce else r_obs
    
    # TFCE Observed
    if use_tfce:
        stat_obs_3d = np.zeros(mask_data.shape, dtype=np.float32)
        stat_obs_3d[mask_data] = stat_obs
        stat_obs_3d = apply_tfce(stat_obs_3d, mask_data, E=tfce_E, H=tfce_H, two_sided=True)
        perm_input_map = stat_obs_3d
    else:
        perm_input_map = stat_obs
    
    # Parallel Permutations
    def _perm_loop(i):
        # Use unique seed for each permutation
        local_rng = np.random.RandomState(42 + i)
        
        # Shuffle behavior
        y_perm = local_rng.permutation(behavior)
        
        if corr_method == 'spearman':
            from scipy.stats import rankdata
            y_perm = rankdata(y_perm)
            
        # Pearson logic
        y_c = y_perm - np.nanmean(y_perm)
        
        # Re-calc norm_y
        norm_y = np.sqrt(np.nansum(y_c**2))
        
        # NaN-safe dot product
        r_perm = np.nansum(X_c * y_c, axis=1) / (norm_x * norm_y)
        
        if use_tfce:
            r_perm = np.clip(r_perm, -0.999999, 0.999999)
            z_perm = np.arctanh(r_perm) # Fisher Z
            
            z_perm_3d = np.zeros(mask_data.shape, dtype=np.float32)
            z_perm_3d[mask_data] = z_perm
            # TFCE
            z_perm_3d = apply_tfce(z_perm_3d, mask_data, E=tfce_E, H=tfce_H, two_sided=True)
            return np.max(np.abs(z_perm_3d))
        else:
            return np.max(np.abs(r_perm))

    null_max = Parallel(n_jobs=-1, verbose=5)(
        delayed(_perm_loop)(i) for i in range(n_perms)
    )
    null_max = np.array(null_max)
    
    # Calculate Parametric Uncorrected P-values
    from scipy.stats import t as t_dist
    df = n_subs - 2
    r_safe_p = np.clip(r_obs, -0.999999, 0.999999)
    t_obs_p = r_safe_p * np.sqrt(df) / np.sqrt(1 - r_safe_p**2)
    p_unc = 2 * (1 - t_dist.cdf(np.abs(t_obs_p), df))

    # Calculate FWER P-values
    if use_tfce:
        obs_flat = perm_input_map[mask_data]
        p_fwer = (np.sum(null_max[:, np.newaxis] >= np.abs(obs_flat[np.newaxis, :]), axis=0) + 1) / (n_perms + 1)
        return r_obs, p_unc, p_fwer
    else:
        # Max-Stat for Standard FWER (if not TFCE but permuted)
        # obs_flat = r_obs
        # p_fwer = (np.sum(null_max[:, np.newaxis] >= np.abs(obs_flat[np.newaxis, :]), axis=0) + 1) / (n_perms + 1)
        return r_obs, p_unc, None # Skipping Max-Stat standard for now unless requested

def extract_seed_name(filename):
    # Extract seed part: "seed-63_-42_9_r5"
    # Stop before _desc
    match = re.search(r'(seed.+?)(?=_desc)', os.path.basename(filename))
    if match:
        return match.group(1)
    return "unknown_seed"

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # 1. Subjects & Behavior
    common_subs, behavior, list1, list2 = load_subjects_and_behavior(
        args.csv_file, args.behavior_col, args.condition, args.contrast_condition
    )
    
    # 2. Load Brain Data
    d1, mask_data, affine = load_brain_data(
        args.input_file, list1, common_subs, config.MASK_FILE, args.roi_id
    )
    
    seed_raw = extract_seed_name(args.input_file)
    
    # Map to friendly name
    SEED_MAPPING = {
        "seed-63_-42_9_r5": "Left_pSTS",
        "seed0_-53_2_r5": "PMC",
        "seed57_-31_5_r5": "Right_pSTS"
    }
    seed_name = SEED_MAPPING.get(seed_raw, seed_raw)
    
    # Organize results by seed
    seed_dir = os.path.join(args.output_dir, seed_name)
    if not os.path.exists(seed_dir):
        os.makedirs(seed_dir)
    args.output_dir = seed_dir
    
    final_data = d1
    
    if args.contrast_file:
        d2, _, _ = load_brain_data(
            args.contrast_file, list2, common_subs, config.MASK_FILE, args.roi_id
        )
        final_data = d1 - d2
        cond_label = f"{args.condition}_vs_{args.contrast_condition}" if args.contrast_condition else "Contrast"
        print(f"Computing Correlation on Contrast: {cond_label}")
    else:
        cond_label = args.condition
        print(f"Computing Correlation on Condition: {cond_label}")
        
    # 3. Correlation & Stats
    r_map, p_unc, p_fwer = run_permutation(
        final_data, behavior, args.n_perms, args.corr_method, 
        mask_data, args.use_tfce, args.tfce_E, args.tfce_H
    )
    
    # 4. Save
    base_name = f"isfc_corr_{cond_label}_{seed_name}_{args.behavior_col}_{args.corr_method}"
    
    if args.use_tfce:
        base_name += "_tfce"
        
    # Save R map
    r_path = os.path.join(args.output_dir, f"{base_name}_desc-r.nii.gz")
    save_map(r_map, mask_data, affine, r_path)
    
    # Plot Raw R
    plot_r_path = os.path.join(args.output_dir, f"{base_name}_desc-r.png")
    save_plot(r_path, plot_r_path, f"Raw Correlation (r): {cond_label} - {seed_name}", positive_only=False)
    
    # Save Uncorrected P map
    p_unc_path = os.path.join(args.output_dir, f"{base_name}_desc-pvalues_uncorrected.nii.gz")
    p_unc_3d = np.ones(mask_data.shape, dtype=np.float32)
    p_unc_3d[mask_data] = p_unc
    save_map(p_unc_3d, mask_data, affine, p_unc_path)
    
    if p_fwer is not None:
        p_final = p_fwer
        p_fwer_path = os.path.join(args.output_dir, f"{base_name}_desc-pvalues.nii.gz")
        p_fwer_3d = np.ones(mask_data.shape, dtype=np.float32)
        p_fwer_3d[mask_data] = p_fwer
        save_map(p_fwer_3d, mask_data, affine, p_fwer_path)
    else:
        p_final = p_unc
    
    # Significant Map (Corrected)
    sig_r = r_map.copy()
    sig_r[p_final >= args.p_threshold] = 0
    sig_r = np.nan_to_num(sig_r)
    
    sig_path = os.path.join(args.output_dir, f"{base_name}_desc-sig_p{str(args.p_threshold).replace('.', '')}.nii.gz")
    save_map(sig_r, mask_data, affine, sig_path)
    
    # Significant Map (Uncorrected) for Plotting comparison
    if args.use_tfce:
        sig_r_unc = r_map.copy()
        sig_r_unc[p_unc >= args.p_threshold] = 0
        sig_r_unc = np.nan_to_num(sig_r_unc)
        
        sig_unc_path = os.path.join(args.output_dir, f"{base_name}_desc-sig_uncorrected.nii.gz")
        sig_r_unc_3d = np.zeros(mask_data.shape, dtype=np.float32)
        sig_r_unc_3d[mask_data] = sig_r_unc
        save_map(sig_r_unc_3d, mask_data, affine, sig_unc_path)
        
        plot_unc_path = os.path.join(args.output_dir, f"{base_name}_desc-sig_uncorrected.png")
        save_plot(sig_unc_path, plot_unc_path, f"Sig Uncorrected p<{args.p_threshold}: {cond_label} {seed_name}", positive_only=False)

    # Plot Corrected
    plot_path = os.path.join(args.output_dir, f"{base_name}_desc-sig.png")
    title = f"Sig Corrected p<{args.p_threshold}: {cond_label} {seed_name}"
    save_plot(sig_path, plot_path, title, positive_only=False)
    
    # NaN Check
    nan_count = np.isnan(r_map).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} voxels have NaN correlation.")

    print(f"Outputs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
