#!/bin/bash
#SBATCH --job-name=ti_isc_analysis
#SBATCH --output=logs/isc_%j.out
#SBATCH --error=logs/isc_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your_email@stanford.edu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Path to your project directory on Sherlock
PROJECT_ROOT="/scratch/users/tongshan/temporal_integration"

# Environment Name (name of the folder where venv is created)
ENV_NAME="isc_env"

# Paths (Override defaults here)
DATA_DIR=""
OUTPUT_DIR="$PROJECT_ROOT/results"
MASK_FILE="$PROJECT_ROOT/TI_code/mask/MNI152_T1_2mm_brain_mask.nii"

# Arguments
CONDITION="TI1_orig"
METHOD="loo"         # loo, pairwise
STATS="phaseshift"   # ttest, bootstrap, phaseshift
PERMS=1000
CHUNK_SIZE=300000    # Set large (e.g. 300000) to disable chunking if you have enough RAM

# ==============================================================================
# EXECUTION
# ==============================================================================

# 1. Load Modules
ml python/3.9.0
ml py-numpy/1.20.3_py39
ml py-scipy/1.6.3_py39
# Add modules required for your brainiak build if needed (e.g., openmpi)
ml openmpi

# 2. Activate Virtual Environment
source $PROJECT_ROOT/code/TI_code/isc_env/bin/activate

# 3. Run Pipeline
# Note: We pass the paths arguments to override the defaults in config.py
# This ensures you don't need to edit python code, just this script.

echo "Starting ISC Analysis for $CONDITION"

python3 $PROJECT_ROOT/code/TI_code/isc/run_isc_pipeline.py \
    --condition $CONDITION \
    --isc_method $METHOD \
    --stats_method $STATS \
    --n_perms $PERMS \
    --chunk_size $CHUNK_SIZE \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mask_file "$MASK_FILE"

echo "Job Finished"
