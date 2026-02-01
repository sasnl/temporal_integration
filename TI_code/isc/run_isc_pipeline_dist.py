import os
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TI_CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if TI_CODE_DIR not in sys.path:
    sys.path.insert(0, TI_CODE_DIR)


from isc import config
print(f"CONFIG MODULE FILE: {os.path.abspath(config.__file__)}", flush=True)

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
    parser.add_argument('--p_threshold', type=float, default=0.05,
                        help='P-value threshold')
    parser.add_argument('--cluster_threshold', type=int, default=0,
                        help='Cluster extent threshold (min voxels). Default: 0')
    parser.add_argument('--use_tfce', action='store_true',
                        help='Use Threshold-Free Cluster Enhancement (requires bootstrap/phaseshift)')
    parser.add_argument('--tfce_E', type=float, default=0.5,
                        help='TFCE extent parameter (default: 0.5)')
    parser.add_argument('--tfce_H', type=float, default=2.0,
                        help='TFCE height parameter (default: 2.0)')
    parser.add_argument("--checkpoint_every", type=int, default=1000,
                        help="Save phaseshift checkpoint every N permutations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume phaseshift from existing checkpoint in output_dir")

    
    # Configurable Paths
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to input data (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help=f'Output directory (default: {config.OUTPUT_DIR})')
    parser.add_argument('--mask_file', type=str, default=config.MASK_FILE,
                        help=f'Path to mask file (default: {config.MASK_FILE})')
    parser.add_argument('--chunk_size', type=int, default=config.CHUNK_SIZE,
                        help='Chunk size for processing (default: 300000)')
    return parser.parse_args()

def run_command(cmd):
    """Run a shell command and check for errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    import time
    start_time = time.time()
    print(f"Pipeline started at: {time.ctime(start_time)}")

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
    print("\n=== STEP 1: ISC COMPUTATION ===")
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
    output_dir = os.path.join(args.output_dir, args.condition)
    os.makedirs(output_dir, exist_ok=True)
    z_map_file = os.path.join(output_dir, f"isc_{args.condition}_{args.isc_method}{roi_suffix}_desc-zscore.nii.gz")
    
    # 2. Run Step 2: Statistics
    print("\n=== STEP 2: ISC STATISTICS ===")
    stats_script = os.path.join(script_dir, 'isc_stats_dist.py')
    cmd_step2 = [
        python_exec, stats_script,
        '--method', args.stats_method,
        '--p_threshold', str(args.p_threshold),
        '--cluster_threshold', str(args.cluster_threshold),
        '--n_perms', str(args.n_perms)
    ] + path_args
    
    if args.use_tfce:
        cmd_step2.extend(['--use_tfce', '--tfce_E', str(args.tfce_E), '--tfce_H', str(args.tfce_H)])
    
    if args.roi_id is not None:
        cmd_step2.extend(['--roi_id', str(args.roi_id)])
        
    if args.stats_method == 'phaseshift':
        cmd_step2.extend(['--condition', args.condition])
        cmd_step2.extend(['--checkpoint_every', str(args.checkpoint_every)])
        cmd_step2.extend(['--isc_method', args.isc_method])
        if args.resume:
            cmd_step2.append('--resume')
    else:
        cmd_step2.extend(['--input_map', z_map_file])

    run_command(cmd_step2)
    
    print("\n=== PIPELINE COMPLETE ===")
    end_time = time.time()
    print(f"Pipeline finished at: {time.ctime(end_time)}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
