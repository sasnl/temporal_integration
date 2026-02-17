import sys
import os

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from pipeline_utils import save_plot

# Target source file (The significant map corresponding to the plot)
# Looking at the file listing, the likely source for "desc-sig.png" is "desc-sig_p005.nii.gz" 
# or generically "desc-sig_p005.nii.gz" is the default significant map.
input_nii = "/Users/tongshan/Documents/TemporalIntegration/result/ISFC/bootstrap/tfce/loo/LpSTS/isfc_TI1_word_loo_seed-63_-42_9_r5_bootstrap_tfce_desc-sig_p005.nii.gz"

# Output file - we overwrite the existing one as requested "re-generate this plot"
output_png = "/Users/tongshan/Documents/TemporalIntegration/result/ISFC/bootstrap/tfce/loo/LpSTS/isfc_TI1_word_loo_seed-63_-42_9_r5_bootstrap_tfce_desc-sig.png"

# Slices
slices = [-40, -28, -16, -6, -4, 6, 22, 32]

print(f"Plotting {input_nii}...")
print(f"Saving to {output_png}...")
print(f"Slices: {slices}")

# Call save_plot
save_plot(
    input_nii, 
    output_png, 
    title="ISFC Word (LpSTS) p<0.05", 
    positive_only=True, # Assuming positive only based on previous context, or standard is mixed? 
    # Actually standard 'sig' plot in isfc_stats.py uses positive_only=True. 
    #   Line 514: save_plot(sig_path, plot_path, ..., positive_only=True)
    # So we keep it True.
    cut_coords=slices
)

print("Done.")
