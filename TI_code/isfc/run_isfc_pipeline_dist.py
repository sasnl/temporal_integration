import argparse
import subprocess
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TI_CODE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if TI_CODE_DIR not in sys.path:
    sys.path.insert(0, TI_CODE_DIR)

from isfc import config
print(f"CONFIG MODULE FILE: {os.path.abspath(config.__file__)}", flush=True)

def run_command(cmd):
    """Run a shell command and check for errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='Run ISFC Analysis Pipeline (Compute + Stats)')
    
    # Common Args
    parser.add_argument('--condition', type=str, required=True, help='Condition name')
    parser.add_argument('--roi_id', type=int, default=None, help='ROI ID (optional)')
    parser.add_argument('--seed_x', type=float, help='Seed X (MNI)')
    parser.add_argument('--seed_y', type=float, help='Seed Y (MNI)')
    parser.add_argument('--seed_z', type=float, help='Seed Z (MNI)')
    parser.add_argument('--seed_radius', type=float, default=5, help='Seed Radius mm')
    parser.add_argument('--seed_file', type=str, help='Seed ROI File (.nii/.nii.gz)')
    
    # Compute Args
    parser.add_argument('--isfc_method', type=str, choices=['loo', 'pairwise'], default='loo', 
                        help='ISFC Method: loo or pairwise')
    
    # Stats Args
    parser.add_argument('--stats_method', type=str, choices=['ttest', 'bootstrap', 'phaseshift'], required=True,
                        help='Statistical Method')
    parser.add_argument('--n_perms', type=int, default=1000, help='Number of permutations/bootstraps')
    parser.add_argument('--p_threshold', type=float, default=0.05, help='P-value threshold')
    parser.add_argument('--cluster_threshold', type=int, default=0,
                        help='Cluster extent threshold (min voxels). Default: 0')
    parser.add_argument('--use_tfce', action='store_true',
                        help='Use Threshold-Free Cluster Enhancement (requires bootstrap/phaseshift)')
    parser.add_argument('--tfce_E', type=float, default=0.5,
                        help='TFCE extent parameter (default: 0.5)')
    parser.add_argument('--tfce_H', type=float, default=2.0,
                        help='TFCE height parameter (default: 2.0)')
    parser.add_argument("--checkpoint_every", type=int, default=25,
                        help="Save phaseshift checkpoint every N permutations")
    parser.add_argument("--resume", action="store_true",
                        help="Resume phaseshift from existing checkpoint in output_dir")
    
    # Configurable Paths
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR,
                        help=f'Path to input data (default: {config.DATA_DIR})')
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR,
                        help=f'Path to output directory (default: {config.OUTPUT_DIR})')
    parser.add_argument('--mask_file', type=str, default=config.MASK_FILE,
                        help=f'Path to mask file (default: {config.MASK_FILE})')
    parser.add_argument('--chunk_size', type=int, default=config.CHUNK_SIZE,
                        help=f'Chunk size (default: {config.CHUNK_SIZE})')
    
    return parser.parse_args()

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
    
    # 1. Run Computation
    print("\n=== STEP 1: ISFC COMPUTATION ===")
    compute_script = os.path.join(script_dir, 'isfc_compute.py')
    
    compute_cmd = [
        python_exec, compute_script,
        "--condition", args.condition,
        "--method", args.isfc_method
    ] + path_args
    
    if args.seed_file:
         compute_cmd.extend(["--seed_file", args.seed_file])
    elif args.seed_x is not None:
         compute_cmd.extend([
            "--seed_x", str(args.seed_x),
            "--seed_y", str(args.seed_y),
            "--seed_z", str(args.seed_z),
            "--seed_radius", str(args.seed_radius)
         ])
    else:
         print("Error: Must provide --seed_file OR --seed_x/y/z")
         sys.exit(1)

    if args.roi_id:
        compute_cmd.extend(["--roi_id", str(args.roi_id)])
        
    run_command(compute_cmd)
    
    # 2. Run Statistics
    print("\n=== STEP 2: ISFC STATISTICS ===")
    
    # Determine input filename from Compute step
    roi_suffix = f"_roi{args.roi_id}" if args.roi_id is not None else ""
    
    if args.seed_file:
         seed_name = f"seed-{os.path.basename(args.seed_file).replace('.nii', '').replace('.gz', '')}"
    else:
         seed_name = f"seed{int(args.seed_x)}_{int(args.seed_y)}_{int(args.seed_z)}_r{int(args.seed_radius)}"
         
    seed_suffix = f"_{seed_name}"
    base_name = f"isfc_{args.condition}_{args.isfc_method}{seed_suffix}{roi_suffix}"
    output_dir = os.path.join(args.output_dir, seed_name)
    # Use Z-score map for T-test/Bootstrap
    input_map = os.path.join(output_dir, f"{base_name}_desc-zscore.nii.gz")
    
    stats_script = os.path.join(script_dir, 'isfc_stats_dist.py')
    stats_cmd = [
        python_exec, stats_script,
        "--method", args.stats_method,
        "--p_threshold", str(args.p_threshold),
        "--n_perms", str(args.n_perms),
        "--cluster_threshold", str(args.cluster_threshold)
    ] + path_args
    
    if args.use_tfce:
        stats_cmd.extend(["--use_tfce", "--tfce_E", str(args.tfce_E), "--tfce_H", str(args.tfce_H)])
    
    if args.stats_method == 'phaseshift':
        # Phase shift specific args
        stats_cmd.extend(["--condition", args.condition])
        
        if args.seed_file:
             stats_cmd.extend(["--seed_file", args.seed_file])
        else:
             stats_cmd.extend([
                "--seed_x", str(args.seed_x),
                "--seed_y", str(args.seed_y),
                "--seed_z", str(args.seed_z),
                "--seed_radius", str(args.seed_radius)
             ])
             
        if args.roi_id:
            stats_cmd.extend(["--roi_id", str(args.roi_id)])
    else:
        # Map-based args
        stats_cmd.extend(["--input_map", input_map])
        if args.roi_id:
             stats_cmd.extend(["--roi_id", str(args.roi_id)])
             
    run_command(stats_cmd)
    
    print("\n=== PIPELINE COMPLETE ===")
    end_time = time.time()
    print(f"Pipeline finished at: {time.ctime(end_time)}")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
