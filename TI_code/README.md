# Temporal Integration (TI) Analysis Code

This repository contains the analysis code for the Temporal Integration project, focusing on Inter-Subject Correlation (ISC) and Inter-Subject Functional Correlation (ISFC) using the [BrainIAK](https://brainiak.org/) library.

The pipeline is designed to be modular, separating computation (Step 1) from statistical analysis (Step 2), and supports High Performance Computing (HPC) usage via parallel processing.

## Project Structure

### Directory Structure Overview

```text
code/TI_code/
├── isc/                  # Inter-Subject Correlation analysis
│   ├── run_isc_pipeline.py
│   ├── isc_compute.py
│   └── isc_stats.py
├── isfc/                 # Inter-Subject Functional Connectivity analysis
│   ├── run_isfc_pipeline.py
│   ├── isfc_compute.py
│   └── isfc_stats.py
├── shared/               # Shared utilities and configuration
│   ├── config.py
│   └── pipeline_utils.py
├── mask/                 # Standard brain masks
│   └── MNI152_T1_2mm_brain_mask.nii
├── batch/                # SLURM batch scripts for HPC
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Dependencies

- Python 3.x
- [BrainIAK](https://brainiak.org/)
- [Nilearn](https://nilearn.github.io/)
- NumPy, SciPy, Joblib
- Matplotlib

Install all dependencies via pip:
```bash
pip install -r requirements.txt
```

## Deployment (Sherlock HPC)

For detailed instructions on deploying and running this code on Stanford's Sherlock cluster, see [Sherlock_Deployment.md](Sherlock_Deployment.md).

## Usage

You can configure input/output paths either by editing `shared/config.py` (recommended for recurring usage) or by passing command-line arguments.

### 1. Inter-Subject Correlation (ISC)

Run the full pipeline using `isc/run_isc_pipeline.py`:

```bash
python isc/run_isc_pipeline.py --condition TI1_orig --isc_method loo --stats_method bootstrap --n_perms 1000
```

**Key Arguments:**
- `--condition`: Name of the experimental condition (e.g., `TI1_orig`).
- `--isc_method`: `loo` (Leave-One-Out) or `pairwise`.
- `--stats_method`: `ttest`, `bootstrap`, or `phaseshift`.
- `--roi_id` (Optional): Run analysis within a specific ROI (using Atlas ID).
- `--p_threshold`: P-value threshold (default: 0.05).

**Path Arguments (Optional overrides):**
- `--data_dir`: Path to input data directory (overrides `config.py`).
- `--output_dir`: Path to output directory (overrides `config.py`).
- `--mask_file`: Path to mask file (overrides `config.py`).
- `--chunk_size`: Number of voxels per chunk (default: 5000). Set to a large number (e.g., 300000) to disable chunking.

### 2. Inter-Subject Functional Correlation (ISFC)

Run the seed-based ISFC pipeline using `isfc/run_isfc_pipeline.py`:

```bash
python isfc/run_isfc_pipeline.py --condition TI1_orig --stats_method phaseshift --seed_x 45 --seed_y -30 --seed_z 10
```

**Key Arguments:**
- `--condition`: Condition name.
- `--seed_x`, `--seed_y`, `--seed_z`: MNI coordinates for the seed.
- `--seed_file`: Path to NIfTI ROI file to use as seed (Mutually exclusive with coordinates).
- `--seed_radius`: Radius of the seed sphere in mm (default: 5).
- `--stats_method`: `ttest`, `bootstrap`, or `phaseshift`.
- `--isfc_method`: `loo` or `pairwise`.

**Path Arguments:**
- `--data_dir`, `--output_dir`, `--mask_file`: Override `config.py` defaults.

## Outputs

Results are saved to `OUTPUT_DIR` (as defined in `config.py` or arguments) with the following naming convention:

- **ISC Maps**: `isc_{condition}_{method}_desc-zscore.nii.gz`
- **Significance Maps**: `..._desc-sig_p005.nii.gz`
- **Plots**: `..._desc-sig.png` (300 DPI, transparent background, positive values only, 'hot' colormap)
