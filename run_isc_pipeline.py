import os
import argparse
import subprocess
import sys
import config

def parse_args():
    parser = argparse.ArgumentParser(description='Run full ISC Analysis Pipeline')
    parser.add_argument('--condition', type=str, required=True, 
                        help='Condition name (e.g., TI1_orig)')
    parser.add_argument('--isc_method', type=str, choices=['loo', 'pairwise'], default='loo',
                        help='ISC method for computation (Step 1)')
    parser.add_argument('--stats_method', type=str, choices=['ttest', 'bootstrap', 'phaseshift'], default='bootstrap',
                        help='Statistical method for Step 2')
    parser.add_argument('--roi_id', type=int, default=None,
                        help='Optional: ROI ID')
    parser.add_argument('--n_perms', type=int, default=1000,
                        help='Number of permutations/bootstraps')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='P-value threshold')
    
    # Configurable Paths
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to input data (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help=f'Output directory (default: {config.OUTPUT_DIR})')
    parser.add_argument('--mask_file', type=str, default=config.MASK_FILE,
                        help=f'Path to mask file (default: {config.MASK_FILE})')
    return parser.parse_args()

def run_command(cmd):
    print(f"\n[Pipeline] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        sys.exit(result.returncode)

def main():
    args = parse_args()
    
    python_exec = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path Arguments to pass down
    path_args = [
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--mask_file', args.mask_file,
        '--chunk_size', str(args.chunk_size)
    ]
    
    # 1. Run Step 1: ISC Computation
    compute_script = os.path.join(script_dir, 'isc_compute.py')
    cmd_step1 = [
        python_exec, compute_script,
        '--condition', args.condition,
        '--method', args.isc_method
    ] + path_args
    
    if args.roi_id is not None:
        cmd_step1.extend(['--roi_id', str(args.roi_id)])
        
    run_command(cmd_step1)
    
    # Define expected output file from Step 1 (needed for Step 2)
    # Naming convention in isc_compute.py:
    # isc_{condition}_{method}{roi_suffix}_desc-zscore.nii.gz
    roi_suffix = f"_roi{args.roi_id}" if args.roi_id is not None else ""
    output_dir = args.output_dir # Use arg instead of hardcode
    z_map_file = os.path.join(output_dir, f"isc_{args.condition}_{args.isc_method}{roi_suffix}_desc-zscore.nii.gz")
    
    # 2. Run Step 2: Statistics
    stats_script = os.path.join(script_dir, 'isc_stats.py')
    cmd_step2 = [
        python_exec, stats_script,
        '--method', args.stats_method,
        '--threshold', str(args.threshold),
        '--n_perms', str(args.n_perms)
    ] + path_args
    
    if args.roi_id is not None:
        cmd_step2.extend(['--roi_id', str(args.roi_id)])
        
    if args.stats_method == 'phaseshift':
        # Phase shift needs condition to reload data
        cmd_step2.extend(['--condition', args.condition])
    else:
        # Map-based methods need the input map
        cmd_step2.extend(['--input_map', z_map_file])
        
    run_command(cmd_step2)
    
    print("\n[Pipeline] Analysis Complete.")

if __name__ == "__main__":
    main()
