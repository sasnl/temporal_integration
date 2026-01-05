# Temporal Integration (TI) Analysis Code

This repository contains the analysis code for the Temporal Integration project, focusing on Inter-Subject Correlation (ISC) and Inter-Subject Functional Correlation (ISFC) using the [BrainIAK](https://brainiak.org/) library.

The pipeline is designed to be modular, separating computation (Step 1) from statistical analysis (Step 2), and supports High Performance Computing (HPC) usage via parallel processing.

## Project Structure

### Pipelines
Orphan scripts to run the full analysis.
- **`run_isc_pipeline.py`**: Orchestrates the ISC analysis (Compute + Stats).
- **`run_isfc_pipeline.py`**: Orchestrates the ISFC analysis (Compute + Stats).

### Core Scripts
- **`isc_compute.py`**: Step 1 of ISC. Computes raw and Fisher-Z transformed ISC maps.
- **`isc_stats.py`**: Step 2 of ISC. Performs statistical tests (T-test, Bootstrap, Phase Shift) on ISC maps.
- **`isfc_compute.py`**: Step 1 of ISFC. Computes seed-based ISFC maps.
- **`isfc_stats.py`**: Step 2 of ISFC. Performs statistical tests on ISFC maps.

### Utilities
- **`isc_utils.py`**: Common helper functions for loading data, masking, and saving NIfTI files.

## Dependencies

- Python 3.x
- [BrainIAK](https://brainiak.org/)
- [Nilearn](https://nilearn.github.io/)
- NumPy, SciPy
- Joblib (for parallelization)

## Usage

### 1. Inter-Subject Correlation (ISC)

Run the full pipeline using `run_isc_pipeline.py`:

```bash
python run_isc_pipeline.py --condition TI1_orig --isc_method loo --stats_method bootstrap --n_perms 1000
```

**Arguments:**
- `--condition`: Name of the experimental condition (e.g., `TI1_orig`).
- `--isc_method`: `loo` (Leave-One-Out) or `pairwise`.
- `--stats_method`: `ttest`, `bootstrap`, or `phaseshift`.
- `--roi_id` (Optional): Run analysis within a specific ROI (using Atlas ID).
- `--threshold`: P-value threshold (default: 0.05).

### 2. Inter-Subject Functional Correlation (ISFC)

Run the seed-based ISFC pipeline using `run_isfc_pipeline.py`:

```bash
python run_isfc_pipeline.py --condition TI1_orig --stats_method phaseshift --seed_x 45 --seed_y -30 --seed_z 10
```

**Arguments:**
- `--condition`: Condition name.
- `--seed_x`, `--seed_y`, `--seed_z`: MNI coordinates for the seed.
- `--seed_radius`: Radius of the seed sphere in mm (default: 5).
- `--stats_method`: `ttest`, `bootstrap`, or `phaseshift`.
- `--roi_id` (Optional): Restrict analysis to an ROI.

## Outputs

Results are saved to the `results/` directory (configured in scripts, defaults to `../result` relative to data or project root typically, check `OUTPUT_DIR` in scripts) with the following naming convention:

- **ISC Maps**: `isc_{condition}_{method}_desc-zscore.nii.gz`
- **Significance Maps**: `..._desc-sig_p005.nii.gz`
- **Plots**: `..._desc-sig.png`
