
import nibabel as nib
import numpy as np
import sys

def check_pvalues(path):
    print(f"Checking {path}")
    try:
        img = nib.load(path)
        data = img.get_fdata()
        # Filter 0s (background) and 1s (default init)
        valid_p = data[(data > 0) & (data < 1.0)]
        
        if valid_p.size == 0:
            print("  No valid p-values found (all 0 or 1).")
            # Check if there are any values != 1 inside mask
            mask = data != 0
            print(f"  Min value in map: {np.min(data)}")
            return

        print(f"  Min p-value: {np.min(valid_p)}")
        print(f"  Max p-value: {np.max(valid_p)}")
        print(f"  Mean p-value: {np.mean(valid_p)}")
        print(f"  # Voxels < 0.05: {np.sum(valid_p < 0.05)}")
        print(f"  # Voxels < 0.001: {np.sum(valid_p < 0.001)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_pvalues(sys.argv[1])
