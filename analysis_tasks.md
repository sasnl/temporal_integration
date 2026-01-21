# Analysis Tasks List for Deployment

This document lists all analysis combinations to be run on Sherlock.

## Overview of Variables

*   **Conditions**: `TI1_orig`, `TI1_sent`, `TI1_word`
*   **ISC Methods**: `loo`, `pairwise`
*   **Stats Method**: `phaseshift`
*   **Correction**: 
    1.  **Standard**: Uncorrected (pixel-wise)
    2.  **TFCE**: FWER-corrected
*   **P-thresholds**: 0.05, 0.01, 0.005, 0.001 (Processed OFFLINE after running once)
*   **ISFC Seeds**:
    1.  **PMC**: `[0, -53, 2]`
    2.  **L-pSTS**: `[-63, -42, 9]`
    3.  **R-pSTS**: `[57, -31, 5]`
*   **Seed Radius**: 5 mm

---

## 1. ISC Analysis
*Total Runs: 3 Conditions × 2 Methods × 2 Correction Types = 12 Jobs*

### Run 1: Standard (No TFCE)
Run these commands to generate standard stats. Offline thresholding can be applied to `_desc-pvalues.nii.gz` later.

**Example Command:**
```bash
python isc/run_isc_pipeline.py \
  --condition TI1_orig \
  --isc_method loo \
  --stats_method phaseshift \
  --n_perms 1000 \
  --p_threshold 0.05
```

**Task Checklist:**
- [ ] `TI1_orig` | `loo` | Standard
- [ ] `TI1_sent` | `loo` | Standard
- [ ] `TI1_word` | `loo` | Standard
- [ ] `TI1_orig` | `pairwise` | Standard
- [ ] `TI1_sent` | `pairwise` | Standard
- [ ] `TI1_word` | `pairwise` | Standard

### Run 2: TFCE (FWER Corrected)
Run these commands to generate TFCE-corrected stats. The script will output an un-thresholded TFCE map (`_desc-tfce.nii.gz`) and a corrected p-value map (`_desc-pvalues.nii.gz`) for offline thresholding.

**Example Command:**
```bash
python isc/run_isc_pipeline.py \
  --condition TI1_orig \
  --isc_method loo \
  --stats_method phaseshift \
  --n_perms 1000 \
  --p_threshold 0.05 \
  --use_tfce
```

**Task Checklist:**
- [ ] `TI1_orig` | `loo` | TFCE
- [ ] `TI1_sent` | `loo` | TFCE
- [ ] `TI1_word` | `loo` | TFCE
- [ ] `TI1_orig` | `pairwise` | TFCE
- [ ] `TI1_sent` | `pairwise` | TFCE
- [ ] `TI1_word` | `pairwise` | TFCE

---

## 2. ISFC Analysis
*Total Runs: 3 Seeds × 3 Conditions × 2 Methods × 2 Correction Types = 36 Jobs*

### Run 1: Standard (No TFCE)

**Example Command (PMC Seed):**
```bash
python isfc/run_isfc_pipeline.py \
  --condition TI1_orig \
  --isfc_method loo \
  --stats_method phaseshift \
  --seed_x 0 --seed_y -53 --seed_z 2 \
  --seed_radius 5 \
  --n_perms 1000 \
  --p_threshold 0.05
```

**Task Checklist (PMC):**
- [ ] `TI1_orig` | `loo` | PMC | Standard
- [ ] `TI1_sent` | `loo` | PMC | Standard
- [ ] `TI1_word` | `loo` | PMC | Standard
- [ ] `TI1_orig` | `pairwise` | PMC | Standard
- [ ] `TI1_sent` | `pairwise` | PMC | Standard
- [ ] `TI1_word` | `pairwise` | PMC | Standard

**Task Checklist (Left pSTS):**
- [ ] `TI1_orig` | `loo` | L-pSTS | Standard
- [ ] `TI1_sent` | `loo` | L-pSTS | Standard
- [ ] `TI1_word` | `loo` | L-pSTS | Standard
- [ ] `TI1_orig` | `pairwise` | L-pSTS | Standard
- [ ] `TI1_sent` | `pairwise` | L-pSTS | Standard
- [ ] `TI1_word` | `pairwise` | L-pSTS | Standard

**Task Checklist (Right pSTS):**
- [ ] `TI1_orig` | `loo` | R-pSTS | Standard
- [ ] `TI1_sent` | `loo` | R-pSTS | Standard
- [ ] `TI1_word` | `loo` | R-pSTS | Standard
- [ ] `TI1_orig` | `pairwise` | R-pSTS | Standard
- [ ] `TI1_sent` | `pairwise` | R-pSTS | Standard
- [ ] `TI1_word` | `pairwise` | R-pSTS | Standard

### Run 2: TFCE (FWER Corrected)

**Example Command (PMC Seed):**
```bash
python isfc/run_isfc_pipeline.py \
  --condition TI1_orig \
  --isfc_method loo \
  --stats_method phaseshift \
  --seed_x 0 --seed_y -53 --seed_z 2 \
  --seed_radius 5 \
  --n_perms 1000 \
  --p_threshold 0.05 \
  --use_tfce
```

**Task Checklist (PMC):**
- [ ] `TI1_orig` | `loo` | PMC | TFCE
- [ ] `TI1_sent` | `loo` | PMC | TFCE
- [ ] `TI1_word` | `loo` | PMC | TFCE
- [ ] `TI1_orig` | `pairwise` | PMC | TFCE
- [ ] `TI1_sent` | `pairwise` | PMC | TFCE
- [ ] `TI1_word` | `pairwise` | PMC | TFCE

**Task Checklist (Left pSTS):**
- [ ] `TI1_orig` | `loo` | L-pSTS | TFCE
- [ ] `TI1_sent` | `loo` | L-pSTS | TFCE
- [ ] `TI1_word` | `loo` | L-pSTS | TFCE
- [ ] `TI1_orig` | `pairwise` | L-pSTS | TFCE
- [ ] `TI1_sent` | `pairwise` | L-pSTS | TFCE
- [ ] `TI1_word` | `pairwise` | L-pSTS | TFCE

**Task Checklist (Right pSTS):**
- [ ] `TI1_orig` | `loo` | R-pSTS | TFCE
- [ ] `TI1_sent` | `loo` | R-pSTS | TFCE
- [ ] `TI1_word` | `loo` | R-pSTS | TFCE
- [ ] `TI1_orig` | `pairwise` | R-pSTS | TFCE
- [ ] `TI1_sent` | `pairwise` | R-pSTS | TFCE
- [ ] `TI1_word` | `pairwise` | R-pSTS | TFCE
