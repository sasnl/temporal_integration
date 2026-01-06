# Sherlock Deployment Guide

This guide explains how to deploy the **TI_code** project to Stanford's Sherlock HPC cluster, set up the Python environment, and run the analysis.

## 1. Transfer Code to Sherlock

## 1. Transfer Code to Sherlock

> [!IMPORTANT]
> **This step must be run from your LOCAL computer's terminal**, NOT from inside Sherlock. 
> If you see `No such file or directory`, you are likely running this on Sherlock by mistake.

1.  **Preparation on Sherlock**: 
    Log in to Sherlock and create a project directory in your scratch space (faster than home).
    ```bash
    # [ON SHERLOCK]
    ssh <your_sunet>@login.sherlock.stanford.edu
    cd $SCRATCH
    mkdir -p temporal_integration
    ```

2.  **Transfer Code**:
    Open a **new** terminal window on your Mac (do not SSH). Run the following to copy the `code` directory using `rsync` (faster and allows excluding files). We exclude `isc_env` because it's Mac-specific.

    ```bash
    # [ON LOCAL MAC]
    rsync -avz --exclude 'isc_env' --exclude '__pycache__' --exclude '.git' --exclude '.DS_Store' /Users/tongshan/Documents/TemporalIntegration/code <your_sunet>@login.sherlock.stanford.edu:/scratch/users/<your_sunet>/temporal_integration/
    ```

3.  **Transfer Data**:
    Similarly, transfer your data folder.

    ```bash
    # [ON LOCAL MAC]
    scp -r /Users/tongshan/Documents/TemporalIntegration/data <your_sunet>@login.sherlock.stanford.edu:/scratch/users/<your_sunet>/temporal_integration/
    ```

## 2. Set Up Environment (One-Time)

On Sherlock, we will create a fresh Python virtual environment. We **cannot** use the one from your Mac.

```bash
# 1. Load Python Module
ml python/3.9.0
ml openmpi  # Required for BrainIAK

# Check that you are using the correct python (should NOT be /bin/python)
which python3

# 2. Go to code directory
cd $SCRATCH/temporal_integration/code/TI_code

# 3. Create Virtual Environment
python3 -m venv isc_env

# 4. Activate it
source isc_env/bin/activate

# 5. Upgrade pip
pip install --upgrade pip

# 6. Install Dependencies
pip install -r requirements.txt
```

> [!TIP]
> **Troubleshooting**: If you see `/bin/python: No module named venv`, it means the `ml python/3.9.0` command wasn't run or didn't work, so the system is trying to use the default (old) system Python. Make sure to run the `ml` commands first!

> **Note:** If `brainiak` fails to install, ensure `openmpi` module is loaded (`ml openmpi`) and you have `mpicc` available.

## 3. Configuration

You have two options to configure paths (Data, Output, Mask):

### Option A: Edit `config.py` (Recommended for default usage)
Edit `code/TI_code/config.py` on Sherlock to match your directories.

```python
DATA_DIR = '/scratch/users/YOUR_SUNET/temporal_integration/data/td/hpf'
OUTPUT_DIR = '/scratch/users/YOUR_SUNET/temporal_integration/results'
MASK_FILE = '/scratch/users/YOUR_SUNET/temporal_integration/code/TI_code/mask/MNI152_T1_2mm_brain_mask.nii'
# ...
```

### Option B: Use Command Line Arguments (Recommended for Batch Jobs)
The scripts now accept `--data_dir`, `--output_dir`, and `--mask_file` arguments which override `config.py`.

## 4. Running Jobs

### Interactive Test
To test quickly, use an interactive session:

```bash
sh_dev -t 1:00:00 # Request 1 hour dev node
ml python/3.9.0 openmpi
source isc_env/bin/activate
python run_isc_pipeline.py --condition TI1_orig
```

### Batch Submission
Edit the `sherlock_job.sbatch` file to point to your correct paths, then submit:

```bash
sbatch sherlock_job.sbatch
```

Monitor your job:
```bash
squeue -u $USER
```

## 5. Sharing & Permissions

By default, your scratch directory is private. To allow collaborators to run your code or view your results, you need to grant them permissions.

### Option A: Grant access to a specific user (Recommended)
Use Access Control Lists (ACLs) to give a specific user read/execute access.

```bash
# 1. Grant traverse permission to your scratch root (so they can reach your folder)
setfacl -m u:<collaborator_sunet>:x /scratch/users/$USER

# 2. Grant read/write/execute permission to your project folder
# -R: Recursive (apply to all files/folders inside)
# -m: Modify ACL
setfacl -R -m u:<collaborator_sunet>:rwx /scratch/users/$USER/temporal_integration

# 3. Ensure future files created in this folder inherit these permissions
setfacl -d -R -m u:<collaborator_sunet>:rwx /scratch/users/$USER/temporal_integration
```

### Option B: Grant access to your group
If your collaborator is in the same unix group as you.

```bash
chmod -R g+rX /scratch/users/$USER/temporal_integration
```
