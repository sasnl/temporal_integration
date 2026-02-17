import sys
import os

# Add shared directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'shared'))
from pipeline_utils import save_plot

# Target file
input_nii = "/Users/tongshan/Documents/TemporalIntegration/result/ISFC_contrast/bootstrap_tfce/loo/Left_pSTS/contrast_TI1_orig_vs_TI1_word_isfc_loo_seed-63_-42_9_r5_bootstrap_tfce_desc-sig_p005.nii.gz"
output_png = input_nii.replace(".nii.gz", "_pos_only.png")

print(f"Plotting {input_nii}...")
print(f"Saving to {output_png}...")

# Call save_plot with positive_only=True
save_plot(
    input_nii, 
    output_png, 
    title="ISFC Contrast (Orig > Word) Positive Only", 
    positive_only=True
)

print("Done.")
