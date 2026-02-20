
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import sys
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed

# Import config and utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from pipeline_utils import load_mask, save_map, save_plot, apply_cluster_threshold, apply_tfce

def parse_args():
    parser = argparse.ArgumentParser(description='Run Paired Comparison of ISC/ISFC Maps (Between Conditions)')
    parser.add_argument('--cond1', type=str, required=True, help='Condition 1 Name (e.g., TI1_orig)')
    parser.add_argument('--cond2', type=str, required=True, help='Condition 2 Name (e.g., TI1_sent)')
    parser.add_argument('--type', type=str, choices=['isc', 'isfc'], required=True, help='Analysis Type')
    parser.add_argument('--method', type=str, choices=['ttest', 'sign_perm', 'bootstrap'], required=True, help='Statistical Method')
    
    parser.add_argument('--seed_name', type=str, default='', help='Seed name string for ISFC file matching (e.g. seed-HG_L). Required if type=isfc.')
    parser.add_argument('--roi_id', type=int, default=None, help='ROI ID for masking')
    parser.add_argument('--n_perms', type=int, default=1000, help='Number of permutations/bootstraps')
    parser.add_argument('--isc_method', type=str, choices=['loo', 'pairwise'], default='loo', help='ISC/ISFC Method (loo vs pairwise). Default: loo')
    
    parser.add_argument('--p_threshold', type=float, default=0.05, help='P-value threshold')
    parser.add_argument('--cluster_threshold', type=int, default=0, help='Cluster threshold (voxels)')
    
    parser.add_argument('--use_tfce', action='store_true', help='Use TFCE (for sign_perm/bootstrap)')
    parser.add_argument('--tfce_E', type=float, default=0.5)
    parser.add_argument('--tfce_H', type=float, default=2.0)
    
    parser.add_argument('--data_dir', type=str, default=config.OUTPUT_DIR, help='Directory containing ISC/ISFC results (Maps)')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help='Output directory')
    parser.add_argument('--subject_list_file', type=str,
                        default='/Users/tongshan/Documents/TemporalIntegration/data/Temporal_integrartion_subjectlist_2026_01_16_updated_filtering.xlsx',
                        help='Path to Subject List Excel file')
    parser.add_argument('--auto_subjects_dir', type=str, default=None,
                        help='Auto-detect subjects from .nii files in this directory (bypasses config and Excel)')

    return parser.parse_args()

def load_subject_intersection(cond1, cond2, excel_path, auto_subjects_dir=None):
    """
    Returns list of subject IDs present in both conditions AND marked as 'have_all_3' in Excel.
    If auto_subjects_dir is set, auto-detect subjects from .nii files instead of config.
    """
    print(f"Loading subject lists...")

    if auto_subjects_dir:
        # Auto-detect subjects from data directory
        import glob as glob_mod
        cond1_dir = os.path.join(auto_subjects_dir, cond1)
        cond2_dir = os.path.join(auto_subjects_dir, cond2)
        files1 = sorted(glob_mod.glob(os.path.join(cond1_dir, '*.nii')))
        files2 = sorted(glob_mod.glob(os.path.join(cond2_dir, '*.nii')))
        list1 = sorted(set(os.path.basename(f).split('_')[0] for f in files1))
        list2 = sorted(set(os.path.basename(f).split('_')[0] for f in files2))

        # Intersection of both conditions (no Excel filtering)
        common = set(list1) & set(list2)
        common_sorted = sorted(list(common))

        print(f"  Auto-detected from {auto_subjects_dir}")
        print(f"  Condition {cond1}: {len(list1)} subjects")
        print(f"  Condition {cond2}: {len(list2)} subjects")
        print(f"  Final Intersection: {len(common_sorted)} subjects")
        print(f"  Subjects: {common_sorted}")
    else:
        # 1. Config lists
        if cond1 not in config.SUBJECT_LISTS or cond2 not in config.SUBJECT_LISTS:
            raise ValueError(f"Conditions {cond1} or {cond2} not found in config.SUBJECT_LISTS")

        list1 = [str(s) for s in config.SUBJECT_LISTS[cond1]]
        list2 = [str(s) for s in config.SUBJECT_LISTS[cond2]]

        # 2. Excel List
        if not os.path.exists(excel_path):
             raise FileNotFoundError(f"Subject list file not found: {excel_path}")

        df = pd.read_excel(excel_path)
        # Filter have_all_3
        if 'have_all_3' not in df.columns or 'PID' not in df.columns:
             raise ValueError("Excel file must contain 'PID' and 'have_all_3' columns")

        valid_subjects = df[df['have_all_3'] == 1]['PID'].astype(str).tolist()

        # 3. Intersection
        common = set(list1) & set(list2) & set(valid_subjects)
        common_sorted = sorted(list(common))

        print(f"  Condition {cond1}: {len(list1)} subjects")
        print(f"  Condition {cond2}: {len(list2)} subjects")
        print(f"  Valid 'have_all_3': {len(valid_subjects)} subjects")
        print(f"  Final Intersection: {len(common_sorted)} subjects")
        print(f"  Subjects: {common_sorted}")

    if len(common_sorted) == 0:
        raise ValueError("No common subjects found!")

    return common_sorted, list1, list2

def load_and_match_data(cond1, cond2, type_str, isc_method, seed_name, common_subs, list1, list2, data_dir, roi_id):
    """
    Load 4D maps and extract matching subjects.
    Returns: data1, data2 (V, N_common), mask_data, mask_affine
    """
    # Construct Filenames
    # Pattern: isc_{cond}_{method}_desc-zscore.nii.gz
    
    roi_suffix = f"_roi{roi_id}" if roi_id is not None else ""
    
    if type_str == 'isc':
        f1 = f"isc_{cond1}_{isc_method}{roi_suffix}_desc-zscore.nii.gz"
        f2 = f"isc_{cond2}_{isc_method}{roi_suffix}_desc-zscore.nii.gz"
    else:
        # ISFC
        f1 = f"isfc_{cond1}_{isc_method}_{seed_name}{roi_suffix}_desc-zscore.nii.gz"
        f2 = f"isfc_{cond2}_{isc_method}_{seed_name}{roi_suffix}_desc-zscore.nii.gz"
        
    p1 = os.path.join(data_dir, f1)
    p2 = os.path.join(data_dir, f2)
    
    if not os.path.exists(p1): raise FileNotFoundError(f"File not found: {p1}")
    if not os.path.exists(p2): raise FileNotFoundError(f"File not found: {p2}")
    
    print(f"Loading Map 1: {f1}")
    img1 = nib.load(p1)
    d1_full = img1.get_fdata(dtype=np.float32)
    affine = img1.affine
    
    print(f"Loading Map 2: {f2}")
    img2 = nib.load(p2)
    d2_full = img2.get_fdata(dtype=np.float32)
    
    # Handle Masking if Maps are 4D full volume (X,Y,Z,S) -> flatten to (V, S)
    # If the file was saved using save_map with a mask, it might already be 4D volume.
    # We should load the standard mask to flatten it, ensuring we only comparing valid voxels.
    mask_data, _ = load_mask(config.MASK_FILE, roi_id)
    
    if d1_full.shape[:3] != mask_data.shape:
        raise ValueError(f"Map dimension mismatch with mask: {d1_full.shape} vs {mask_data.shape}")
        
    # Extract indices
    # list1 is the order in d1_full, list2 is the order in d2_full
    idx1 = [list1.index(s) for s in common_subs]
    idx2 = [list2.index(s) for s in common_subs]
    
    # Flatten and Select
    # d1_full[mask_data] -> (V, S_total)
    d1_flat = d1_full[mask_data][:, idx1] # (V, N_common)
    d2_flat = d2_full[mask_data][:, idx2] # (V, N_common)
    
    return d1_flat, d2_flat, mask_data, affine

def _run_signperm_iter(i, diff_data, use_tfce, mask_3d, tfce_E, tfce_H, seed):
    rng = np.random.RandomState(seed)
    n_samples = diff_data.shape[1]
    
    # Random signs: +1 or -1
    signs = rng.choice([-1, 1], size=n_samples)
    
    # Apply signs (broadcast)
    perm_diff = diff_data * signs[np.newaxis, :]
    perm_mean = np.mean(perm_diff, axis=1)
    
    if use_tfce:
        # TFCE on the mean map
        perm_mean_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        perm_mean_3d[mask_3d] = perm_mean
        perm_mean_3d = apply_tfce(perm_mean_3d, mask_3d, E=tfce_E, H=tfce_H, two_sided=True)
        return np.max(np.abs(perm_mean_3d))
    else:
        return perm_mean

def run_sign_permutation(diff_data, n_perms, mask_data, use_tfce, tfce_E, tfce_H):
    print(f"Running Sign Permutation (n={n_perms})...")
    
    obs_mean = np.nanmean(diff_data, axis=1)
    
    if use_tfce:
         obs_mean_3d = np.zeros(mask_data.shape, dtype=np.float32)
         obs_mean_3d[mask_data] = obs_mean
         obs_mean_3d = apply_tfce(obs_mean_3d, mask_data, E=tfce_E, H=tfce_H, two_sided=True)
         obs_metric = obs_mean_3d[mask_data] # Flatten back for comparison? Or keep 3D?
         # Correction compares Obs(v) vs Max(Null)
    else:
         obs_metric = obs_mean

    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_signperm_iter)(
            i, diff_data, use_tfce, mask_data, tfce_E, tfce_H, 42 + i
        ) for i in range(n_perms)
    )
    
    if use_tfce:
        null_max_stats = np.array(results) # (n_perms,)
        # FWER P-value
        p_values = (np.sum(null_max_stats[np.newaxis, :] >= np.abs(obs_metric[:, np.newaxis]), axis=1) + 1) / (n_perms + 1)
        mean_map = obs_metric # The TFCE enhanced map
    else:
        # Voxelwise
        null_means = np.array(results).T # (V, n_perms)
        p_values = (np.sum(np.abs(null_means) >= np.abs(obs_metric[:, np.newaxis]), axis=1) + 1) / (n_perms + 1)
        mean_map = obs_metric

    return mean_map, p_values

def _run_bootstrap_iter(i, diff_data, use_tfce, mask_3d, tfce_E, tfce_H, seed):
    rng = np.random.RandomState(seed)
    n_samples = diff_data.shape[1]
    
    # Resample subjects with replacement
    indices = rng.randint(0, n_samples, size=n_samples)
    sample = diff_data[:, indices]
    boot_mean = np.mean(sample, axis=1)
    
    if use_tfce:
        boot_mean_3d = np.zeros(mask_3d.shape, dtype=np.float32)
        boot_mean_3d[mask_3d] = boot_mean
        boot_mean_3d = apply_tfce(boot_mean_3d, mask_3d, E=tfce_E, H=tfce_H, two_sided=True)
        return np.max(np.abs(boot_mean_3d))
    else:
        return boot_mean

def run_bootstrap_contrast(diff_data, n_perms, mask_data, use_tfce, tfce_E, tfce_H):
    # Bootstrap for contrast is tricky.
    # Usually: Test if CI overlaps 0.
    # Or: Shift Null distribution?
    # Standard: Test statistic = Mean / SE.
    # Or: "Bootstrap Hypothesis Testing" involves centering the data first to create a null, then resampling.
    # However, if the user just wants "Bootstrap", it usually implies assessing stability.
    # Given the previous context, we'll assume a Null-Hypothesis test using Centered Bootstrap (Hall & Wilson) or just simple P-value from distribution crossing 0?
    
    # Let's use the same "Null Hypothesis" approach as `isc_stats.py`:
    # 1. Obseved Mean
    # 2. Shift data to have mean 0: Data_null = Data - Mean(Data)
    # 3. Bootstrap Data_null to generate Null Distribution of Means
    # 4. Compare Observed Mean to Null Means.
    
    print(f"Running Bootstrap Test (n={n_perms})...")
    
    obs_mean = np.nanmean(diff_data, axis=1)
    
    if use_tfce:
        obs_mean_3d = np.zeros(mask_data.shape, dtype=np.float32)
        obs_mean_3d[mask_data] = obs_mean
        obs_mean_3d = apply_tfce(obs_mean_3d, mask_data, E=tfce_E, H=tfce_H, two_sided=True)
        obs_metric = obs_mean_3d[mask_data]
    else:
        obs_metric = obs_mean
        
    # Center the data to enforce H0: mean=0
    # Center each subject? No, Center the group distribution.
    # New Data = Data - GrandMean. So Mean(New Data) = 0.
    grand_mean = np.nanmean(diff_data, axis=1, keepdims=True)
    null_data = diff_data - grand_mean
    
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_bootstrap_iter)(
            i, null_data, use_tfce, mask_data, tfce_E, tfce_H, 42 + i
        ) for i in range(n_perms)
    )
    
    if use_tfce:
        null_max_stats = np.array(results) 
        p_values = (np.sum(null_max_stats[np.newaxis, :] >= np.abs(obs_metric[:, np.newaxis]), axis=1) + 1) / (n_perms + 1)
        mean_map = obs_metric
    else:
        null_means = np.array(results).T
        p_values = (np.sum(np.abs(null_means) >= np.abs(obs_metric[:, np.newaxis]), axis=1) + 1) / (n_perms + 1)
        mean_map = obs_metric
        
    return mean_map, p_values


def main():
    args = parse_args()
    
    # 1. Subjects
    common_subs, list1, list2 = load_subject_intersection(args.cond1, args.cond2, args.subject_list_file, auto_subjects_dir=args.auto_subjects_dir)
    
    # 2. Load Data
    data1, data2, mask_data, mask_affine = load_and_match_data(
        args.cond1, args.cond2, args.type, args.isc_method, args.seed_name, 
        common_subs, list1, list2, args.data_dir, args.roi_id
    )
    
    # 3. Compute Difference
    # Note: data are Z-scores. Diff = Z1 - Z2.
    diff_data = data1 - data2
    
    # 4. Statistics
    if args.method == 'ttest':
        if args.use_tfce:
            print("Warning: T-test ignores TFCE.")
        print("Running Paired T-test...")
        t_stats, p_values = ttest_1samp(diff_data, popmean=0, axis=1, nan_policy='omit')
        mean_map = np.nanmean(diff_data, axis=1) # Valid to save mean diff or T-stat? Usually save T-stat?
        # Let's save Mean Diff for consistency with "Contrast Magnitude"
        # But maybe T-stat is more informative?
        # Standard: Save "Stat" map. For t-test, it's T. For others, it's Mean/TFCE.
        # Let's save Mean Diff as the primary "effect size" map, similar to ISC/ISFC maps.
    
    elif args.method == 'sign_perm':
        mean_map, p_values = run_sign_permutation(
            diff_data, args.n_perms, mask_data, args.use_tfce, args.tfce_E, args.tfce_H
        )
        
    elif args.method == 'bootstrap':
        mean_map, p_values = run_bootstrap_contrast(
            diff_data, args.n_perms, mask_data, args.use_tfce, args.tfce_E, args.tfce_H
        )
        
    # 5. Save Outputs
    base_name = f"contrast_{args.cond1}_vs_{args.cond2}_{args.type}_{args.isc_method}"
    if args.seed_name: base_name += f"_{args.seed_name}"
    base_name += f"_{args.method}"
    if args.roi_id: base_name += f"_roi{args.roi_id}"
    if args.use_tfce: base_name += "_tfce"
    
    # Stat Map
    stat_path = os.path.join(args.output_dir, f"{base_name}_desc-stat.nii.gz")
    save_map(mean_map, mask_data, mask_affine, stat_path)
    
    # P Values
    p_path = os.path.join(args.output_dir, f"{base_name}_desc-pvalues.nii.gz")
    # p_values is (V,). Need to map to 3D
    p_values_3d = np.ones(mask_data.shape, dtype=np.float32)
    p_values_3d[mask_data] = p_values
    save_map(p_values_3d, mask_data, mask_affine, p_path)
    
    # Significant Map
    sig_map = mean_map.copy()
    sig_map[p_values >= args.p_threshold] = 0
    
    if args.cluster_threshold > 0:
        # Reconstruct 3D for clustering
        sig_map_3d = np.zeros(mask_data.shape, dtype=np.float32)
        sig_map_3d[mask_data] = sig_map
        sig_map_3d = apply_cluster_threshold(sig_map_3d, args.cluster_threshold)
        sig_map = sig_map_3d[mask_data] # Flatten back if we want? Actually save_map handles 3D.
        mean_map_to_save = sig_map_3d
    else:
        mean_map_to_save = np.zeros(mask_data.shape, dtype=np.float32)
        mean_map_to_save[mask_data] = sig_map
        
    sig_path = os.path.join(args.output_dir, f"{base_name}_desc-sig_p{str(args.p_threshold).replace('.', '')}.nii.gz")
    save_map(mean_map_to_save, mask_data, mask_affine, sig_path)
    
    # Plot
    plot_path = os.path.join(args.output_dir, f"{base_name}_desc-sig.png")
    save_plot(sig_path, plot_path, f"Contrast {args.cond1} vs {args.cond2} ({args.method})", positive_only=False)
    
    print("Contrast Analysis Complete.")
    print(f"Outputs:\n {stat_path}\n {p_path}\n {sig_path}")

if __name__ == "__main__":
    main()
