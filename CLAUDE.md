# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Temporal Integration (TI) neuroimaging analysis pipeline for computing Inter-Subject Correlation (ISC) and Inter-Subject Functional Connectivity (ISFC) using BrainIAK. Designed for both local execution and Stanford Sherlock HPC cluster.

## Environment Setup

```bash
python3 -m venv isc_env
source isc_env/bin/activate
pip install -r requirements.txt
```

Python 3.12 locally, 3.9 on Sherlock. The `isc_env/` virtualenv is gitignored.

## Running Analyses

All commands run from the `TI_code/` root directory.

**ISC pipeline (single run):**
```bash
python isc/run_isc_pipeline.py --condition TI1_orig --isc_method loo --stats_method phaseshift --n_perms 1000 --use_tfce
```

**ISFC pipeline (single run):**
```bash
python isfc/run_isfc_pipeline.py --condition TI1_orig --isfc_method loo --stats_method phaseshift --seed_x 0 --seed_y -53 --seed_z 2 --seed_radius 5 --n_perms 1000 --use_tfce
```

**Condition contrasts:**
```bash
python shared/run_contrast.py --cond1 TI1_orig --cond2 TI1_sent --type isc --isc_method loo --method bootstrap --n_perms 1000 --use_tfce
```

**Re-threshold existing results (no recomputation):**
```bash
python shared/rethreshold_results.py --result_dir /path/to/results --thresholds 0.01 0.005 0.001
```

**Batch shell scripts** in `scripts/` run all condition/method/seed combinations:
- `scripts/run_isc_bootstrap_local.sh`, `scripts/run_isc_fdr_local.sh` — ISC batch runs
- `scripts/run_isfc_bootstrap_local.sh`, `scripts/run_isfc_ttest_local.sh` — ISFC batch runs
- `scripts/run_isc_contrast_commands.sh`, `scripts/run_isfc_contrast_commands.sh` — contrast batch runs

**HPC array jobs:**
```bash
python batch/generate_params.py   # generates parameter files
sbatch batch/run_isc_array.sbatch  # submits 12 ISC jobs
sbatch batch/run_isfc_array.sbatch # submits 36 ISFC jobs
```

## Architecture

### Two-Stage Pipeline Design

Every analysis separates **computation** (Step 1) from **statistics** (Step 2):

- **Step 1 — Compute**: Load fMRI data, compute ISC/ISFC correlations, apply Fisher-Z transform, save 4D maps (`_desc-zscore.nii.gz`, `_desc-raw.nii.gz`)
- **Step 2 — Stats**: Load Z-score maps, run statistical tests (t-test/bootstrap/phase-shift), apply correction (TFCE/max-stat/FDR/cluster), save significance maps

This separation allows re-running statistics with different parameters without recomputing correlations.

### Module Layout

| Directory | Purpose |
|-----------|---------|
| `isc/` | ISC pipeline: `run_isc_pipeline.py` (orchestrator) → `isc_compute.py` (Step 1) → `isc_stats.py` (Step 2) |
| `isfc/` | ISFC pipeline: same pattern, plus behavioral correlation |
| `shared/` | `config.py` (paths, subject lists), `pipeline_utils.py` (core utilities: data loading, mask handling, chunked ISC, TFCE, NIfTI I/O), `run_contrast.py` (paired condition comparisons), `rethreshold_results.py` |
| `batch/` | All SLURM/HPC scripts: array job templates, single-job templates, and parameter generation |
| `scripts/` | Local batch runner shell scripts for running all condition/method/seed combinations |
| `utils/` | One-time data exploration and QC scripts (`check_pvals.py`, `extract_subjects.py`, `inspect_excel.py`, `process_demographics.py`) |
| `mask/` | MNI152 2mm brain mask |

### Configuration

`shared/config.py` defines data paths, output paths, mask path, and subject lists per condition. All can be overridden via CLI arguments (`--data_dir`, `--output_dir`, `--mask_file`, `--chunk_size`).

### Key Domain Concepts

**Three conditions:** `TI1_orig` (original), `TI1_sent` (sentence-scrambled), `TI1_word` (word-shuffled)

**ISC methods:** `loo` (leave-one-out) or `pairwise`

**Statistical methods:** `ttest` (uncorrected), `bootstrap` (resampling), `phaseshift` (temporal randomization preserving autocorrelation)

**Correction methods:** TFCE (recommended, FWER-corrected), max-stat FWE, cluster thresholding, FDR. TFCE and cluster threshold are mutually exclusive. TFCE requires bootstrap or phaseshift.

**ISFC seeds:** PMC `[0,-53,2]`, L-pSTS `[-63,-42,9]`, R-pSTS `[57,-31,5]` — all 5mm radius spheres.

**Subject filtering for contrasts:** Uses `have_all_3` column from Excel demographics file to ensure paired design integrity.

### Output Naming Convention

```
{isc|isfc}_{condition}_{method}[_seed]_desc-{descriptor}.nii.gz
```
Descriptors: `zscore` (4D correlations), `raw` (4D), `meanz` (3D mean), `stat` (3D test statistic), `tfce` (3D TFCE-enhanced), `pvalues` (3D), `sig_p{thresh}` (3D thresholded)

### Data Flow

```
Input: /data/ti_processed/{condition}/*.nii (preprocessed fMRI per subject)
  → Computation: Fisher-Z ISC/ISFC maps (4D, one volume per subject/pair)
  → Statistics: test statistic + p-value maps (3D)
  → Correction: TFCE/FWE/cluster-corrected significance maps (3D)
Output: /result/{ISC|ISFC|ISC_contrast}/...
```

### Processing Details

- Chunked parallel computation via Joblib (`chunk_size=5000` voxels default)
- Phase-shift supports checkpointing (`--checkpoint_every 25`, `--resume`)
- `pipeline_utils.py` contains all shared logic: `load_data()`, `run_isc_computation()`, `apply_tfce()`, `save_map()`, `save_plot()`
