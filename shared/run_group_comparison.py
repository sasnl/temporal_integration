import os
import argparse
import numpy as np
import nibabel as nib
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from pipeline_utils import load_mask, save_map, save_plot, apply_tfce
from joblib import Parallel, delayed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Group Comparison: NT vs ASD (Two-Sample Permutation Test)'
    )
    parser.add_argument('--condition', type=str, required=True,
                        help='Condition name (e.g., TI1_orig)')
    parser.add_argument('--analysis_type', type=str, choices=['isc', 'isfc'],
                        required=True, help='Analysis type: isc or isfc')
    parser.add_argument('--method', type=str, choices=['loo', 'pairwise'],
                        default='loo', help='ISC/ISFC method (default: loo)')
    parser.add_argument('--seed_name', type=str, default='',
                        help='Seed name for ISFC (e.g., seed0_-53_2_r5)')
    parser.add_argument('--nt_result_dir', type=str, required=True,
                        help='Directory containing NT Z-score maps')
    parser.add_argument('--asd_result_dir', type=str, required=True,
                        help='Directory containing ASD Z-score maps')
    parser.add_argument('--output_dir', type=str, default=config.GROUP_OUTPUT_DIR,
                        help='Output directory for group comparison results')
    parser.add_argument('--mask_file', type=str, default=config.MASK_FILE,
                        help='Path to brain mask')
    parser.add_argument('--n_perms', type=int, default=1000,
                        help='Number of permutations (default: 1000)')
    parser.add_argument('--p_threshold', type=float, default=0.05,
                        help='Significance threshold (default: 0.05)')
    parser.add_argument('--tfce_E', type=float, default=0.5,
                        help='TFCE extent parameter')
    parser.add_argument('--tfce_H', type=float, default=2.0,
                        help='TFCE height parameter')
    parser.add_argument('--tfce_dh', type=float, default=0.01,
                        help='TFCE step size')
    return parser.parse_args()


def build_zscore_filename(analysis_type, condition, method, seed_name=''):
    """Build the Z-score map filename matching existing naming conventions."""
    if analysis_type == 'isc':
        return f"isc_{condition}_{method}_desc-zscore.nii.gz"
    else:
        return f"isfc_{condition}_{method}_{seed_name}_desc-zscore.nii.gz"


def load_group_data(zscore_path, mask):
    """
    Load a 4D Z-score map and extract masked voxels.
    Returns: data (n_voxels, n_subjects)
    """
    if not os.path.exists(zscore_path):
        raise FileNotFoundError(f"Z-score map not found: {zscore_path}")

    print(f"Loading: {zscore_path}")
    img = nib.load(zscore_path)
    data_4d = img.get_fdata(dtype=np.float32)

    if data_4d.shape[:3] != mask.shape:
        raise ValueError(
            f"Shape mismatch: map {data_4d.shape[:3]} vs mask {mask.shape}"
        )

    n_subjects = data_4d.shape[3]
    data_masked = data_4d[mask]
    print(f"  Loaded {n_subjects} subjects, {data_masked.shape[0]} voxels")
    return data_masked


def _run_perm_iter(i, pooled_data, n_nt, n_asd, use_tfce, mask, tfce_E, tfce_H, tfce_dh, seed):
    """Single permutation iteration: shuffle group labels, compute difference."""
    rng = np.random.RandomState(seed)
    n_total = n_nt + n_asd

    perm_indices = rng.permutation(n_total)
    pseudo_nt = pooled_data[:, perm_indices[:n_nt]]
    pseudo_asd = pooled_data[:, perm_indices[n_nt:]]

    perm_diff = np.nanmean(pseudo_nt, axis=1) - np.nanmean(pseudo_asd, axis=1)

    if use_tfce:
        perm_diff_3d = np.zeros(mask.shape, dtype=np.float32)
        perm_diff_3d[mask] = perm_diff
        perm_diff_3d = apply_tfce(perm_diff_3d, mask, E=tfce_E, H=tfce_H,
                                   dh=tfce_dh, two_sided=True)
        return np.max(np.abs(perm_diff_3d))
    else:
        return perm_diff


def run_two_sample_permutation(nt_data, asd_data, n_perms, mask,
                                tfce_E, tfce_H, tfce_dh, use_tfce=True):
    """
    Two-sample permutation test comparing NT vs ASD.

    Returns:
        obs_diff: observed mean difference (1D, n_voxels)
        p_values: two-sided p-values (1D, n_voxels)
        obs_tfce_3d: TFCE-enhanced observed map (3D) or None
    """
    n_nt = nt_data.shape[1]
    n_asd = asd_data.shape[1]
    print(f"Group comparison: NT (n={n_nt}) vs ASD (n={n_asd})")
    print(f"Permutations: {n_perms}, TFCE: {use_tfce}")

    obs_diff = np.nanmean(nt_data, axis=1) - np.nanmean(asd_data, axis=1)

    pooled_data = np.concatenate([nt_data, asd_data], axis=1)

    obs_tfce_3d = None
    if use_tfce:
        obs_diff_3d = np.zeros(mask.shape, dtype=np.float32)
        obs_diff_3d[mask] = obs_diff
        obs_tfce_3d = apply_tfce(obs_diff_3d, mask, E=tfce_E, H=tfce_H,
                                  dh=tfce_dh, two_sided=True)
        obs_metric = obs_tfce_3d[mask]
    else:
        obs_metric = obs_diff

    print(f"Running {n_perms} permutations...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(_run_perm_iter)(
            i, pooled_data, n_nt, n_asd, use_tfce, mask,
            tfce_E, tfce_H, tfce_dh, 42 + i
        ) for i in range(n_perms)
    )

    if use_tfce:
        null_max_stats = np.array(results)
        p_values = (
            np.sum(null_max_stats[np.newaxis, :] >=
                   np.abs(obs_metric[:, np.newaxis]), axis=1) + 1
        ) / (n_perms + 1)
    else:
        null_diffs = np.array(results).T
        p_values = (
            np.sum(np.abs(null_diffs) >=
                   np.abs(obs_metric[:, np.newaxis]), axis=1) + 1
        ) / (n_perms + 1)

    return obs_diff, p_values, obs_tfce_3d


def main():
    args = parse_args()
    start_time = time.time()

    print("=" * 60)
    print("GROUP COMPARISON: NT vs ASD")
    print("=" * 60)
    print(f"Condition:     {args.condition}")
    print(f"Analysis:      {args.analysis_type}")
    print(f"Method:        {args.method}")
    if args.seed_name:
        print(f"Seed:          {args.seed_name}")
    print(f"Permutations:  {args.n_perms}")
    print(f"TFCE:          E={args.tfce_E}, H={args.tfce_H}, dh={args.tfce_dh}")
    print(f"NT results:    {args.nt_result_dir}")
    print(f"ASD results:   {args.asd_result_dir}")
    print()

    correction_dir = "tfce"
    out_dir = os.path.join(args.output_dir, correction_dir, args.method)
    os.makedirs(out_dir, exist_ok=True)

    mask, affine = load_mask(args.mask_file)

    zscore_fname = build_zscore_filename(
        args.analysis_type, args.condition, args.method, args.seed_name
    )
    nt_path = os.path.join(args.nt_result_dir, zscore_fname)
    asd_path = os.path.join(args.asd_result_dir, zscore_fname)

    nt_data = load_group_data(nt_path, mask)
    asd_data = load_group_data(asd_path, mask)

    obs_diff, p_values, obs_tfce_3d = run_two_sample_permutation(
        nt_data, asd_data, args.n_perms, mask,
        args.tfce_E, args.tfce_H, args.tfce_dh, use_tfce=True
    )

    # --- Save outputs ---
    base_name = f"group_nt_vs_asd_{args.condition}_{args.analysis_type}_{args.method}"
    if args.seed_name:
        base_name += f"_{args.seed_name}"

    stat_path = os.path.join(out_dir, f"{base_name}_desc-stat.nii.gz")
    save_map(obs_diff, mask, affine, stat_path)

    if obs_tfce_3d is not None:
        tfce_path = os.path.join(out_dir, f"{base_name}_desc-tfce.nii.gz")
        save_map(obs_tfce_3d, mask, affine, tfce_path)

    p_3d = np.ones(mask.shape, dtype=np.float32)
    p_3d[mask] = p_values
    p_path = os.path.join(out_dir, f"{base_name}_desc-pvalues.nii.gz")
    save_map(p_3d, mask, affine, p_path)

    sig_data = obs_diff.copy()
    sig_data[p_values >= args.p_threshold] = 0
    sig_path = os.path.join(
        out_dir,
        f"{base_name}_desc-sig_p{str(args.p_threshold).replace('.', '')}.nii.gz"
    )
    save_map(sig_data, mask, affine, sig_path)

    plot_path = os.path.join(out_dir, f"{base_name}_desc-sig.png")
    title = f"Group NT vs ASD: {args.condition} ({args.analysis_type} {args.method})"
    save_plot(sig_path, plot_path, title, positive_only=False)

    n_sig = np.sum(p_values < args.p_threshold)
    n_total = np.sum(mask)
    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Significant voxels: {n_sig} / {n_total} "
          f"({100 * n_sig / n_total:.2f}%) at p < {args.p_threshold}")
    print(f"Min p-value: {np.min(p_values):.4f}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Outputs:")
    print(f"  Stat map:    {stat_path}")
    print(f"  P-values:    {p_path}")
    print(f"  Sig map:     {sig_path}")
    print(f"  Plot:        {plot_path}")


if __name__ == "__main__":
    main()
