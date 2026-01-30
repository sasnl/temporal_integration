
import os
import shutil
import re
import glob

# Path to results
result_dir = 'result/ISFC_behav'
files = glob.glob(os.path.join(result_dir, 'isfc_corr_*.nii.gz')) + glob.glob(os.path.join(result_dir, 'isfc_corr_*.png'))

print(f"Found {len(files)} files to organize.")

# Regex to extract seed
# Filenames: isfc_corr_..._seed-..._srs...
# Or: isfc_corr_..._seed-..._desc-zscore...
# We rely on the pattern 'seed' until next '_' or special boundary?
# Wait, seed names contain underscores: seed-63_-42_9_r5.
# But they are usually followed by _srs or _desc.
# Regex: (seed.+?)_(srs|desc)

count = 0
for f in files:
    fname = os.path.basename(f)
    if os.path.isdir(f): continue
    
    # Try match
    match = re.search(r'(seed.+?)_(srs|desc)', fname)
    if match:
        seed_name = match.group(1)
        
        # Cleanup regex might grab trailing _desc if not careful
        # If seed name has _desc in it? No, files usually: 
        # isfc_corr_TI1_orig_seed-63_-42_9_r5_desc-zscore...
        # Matched 'seed-63_-42_9_r5' before '_desc'.
        # isfc_corr_TI1_orig_seed-63_-42_9_r5_srs...
        # Matched 'seed-63_-42_9_r5' before '_srs'.
        
        # Double check if seed_name ends with _r5 or similar
        # It should process correctly.
        
        target_dir = os.path.join(result_dir, seed_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        shutil.move(f, os.path.join(target_dir, fname))
        count += 1
    else:
        print(f"Skipping (no seed match): {fname}")

print(f"Moved {count} files.")
