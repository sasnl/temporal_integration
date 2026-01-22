
import os
import glob
import numpy as np
import nibabel as nib
import sys
import argparse

# Since this script is now in 'shared', we can import directly
try:
    from pipeline_utils import save_map, save_plot
except ImportError:
    # If run from outside shared (e.g., from root), need to adjust path or assume shared is in pythonpath
    # But usually this is run as 'python shared/rethreshold_results.py'
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from pipeline_utils import save_map, save_plot

def rethreshold_results(result_dir, thresholds):
    """
    Scan result directory for p-value maps (ISC and ISFC) and apply additional thresholds.
    """
    print(f"Scanning directory: {result_dir}")
    print(f"Applying thresholds: {thresholds}")
    
    # Find all p-value maps
    # Pattern: *_desc-pvalues.nii.gz
    # This pattern is used by both ISC and ISFC stats scripts.
    count = 0
    for root, dirs, files in os.walk(result_dir):
        for file in files:
            if file.endswith('_desc-pvalues.nii.gz'):
                p_map_path = os.path.join(root, file)
                base_name = file.replace('_desc-pvalues.nii.gz', '')
                
                # Verify corresponding stat file exists
                # ISC/ISFC naming: _desc-stat.nii.gz or _desc-tfce.nii.gz
                stat_path = os.path.join(root, f"{base_name}_desc-stat.nii.gz")
                if not os.path.exists(stat_path):
                    stat_path = os.path.join(root, f"{base_name}_desc-tfce.nii.gz")
                    if not os.path.exists(stat_path):
                        print(f"Skipping {base_name}: No matching stat/tfce map found.")
                        continue
                
                print(f"Processing: {base_name}")
                count += 1
                
                # Load maps
                img_p = nib.load(p_map_path)
                p_values = img_p.get_fdata()
                affine = img_p.affine
                mask = np.ones(p_values.shape, dtype=bool) 
                
                img_stat = nib.load(stat_path)
                stat_values = img_stat.get_fdata()
                
                for thresh in thresholds:
                    thresh_str = str(thresh).replace('.', '')
                    # Handle both isc_ and isfc_ prefixes if needed, but base_name captures it.
                    # Naming convention: {base}_desc-sig_p{thresh}.nii.gz
                    output_sig_path = os.path.join(root, f"{base_name}_desc-sig_p{thresh_str}.nii.gz")
                    
                    if os.path.exists(output_sig_path):
                        print(f"  Skipping p<{thresh}: already exists.")
                        continue
                        
                    print(f"  Applying threshold p<{thresh}")
                    
                    # Create significant map
                    sig_map = stat_values.copy()
                    sig_map[p_values >= thresh] = 0
                    
                    # Save
                    save_map(sig_map, mask, affine, output_sig_path)
                    
                    # Plot
                    output_plot_path = os.path.join(root, f"{base_name}_desc-sig_p{thresh_str}.png")
                    # Determine title type (ISC or ISFC)
                    analysis_type = "ISFC" if "isfc" in base_name else "ISC"
                    save_plot(output_sig_path, output_plot_path, f"Sig {analysis_type} (p<{thresh})", positive_only=True)
    
    print(f"Finished. Processed {count} maps.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Re-threshold ISC/ISFC results.')
    parser.add_argument('--result_dir', type=str, required=True, 
                        help='Root directory containing results (e.g., /path/to/result)')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.01, 0.005, 0.001],
                        help='List of p-value thresholds to apply (default: 0.01 0.005 0.001)')
    args = parser.parse_args()
    
    rethreshold_results(args.result_dir, args.thresholds)
