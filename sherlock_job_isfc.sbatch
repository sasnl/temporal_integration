#!/bin/bash
#SBATCH --job-name=ti_isfc_analysis
#SBATCH --output=logs/isfc_%j.out
#SBATCH --error=logs/isfc_%j.err
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
ANALYSIS_METHOD="loo"     # loo, pairwise
STATS_METHOD="phaseshift" # ttest, bootstrap, phaseshift
N_PERMS=1000
CHUNK_SIZE=300000         # Set large to disable chunking if memory permits

# Seed Definition (MNI Coordinates)
SEED_X=45
SEED_Y=-30
SEED_Z=10
SEED_RADIUS=5

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
echo "Starting ISFC Analysis for $CONDITION"
echo "Seed: ($SEED_X, $SEED_Y, $SEED_Z), Radius: $SEED_RADIUS"

python3 $PROJECT_ROOT/code/TI_code/run_isfc_pipeline.py \
    --condition $CONDITION \
    --analysis_method $ANALYSIS_METHOD \
    --stats_method $STATS_METHOD \
    --n_perms $N_PERMS \
    --seed_x $SEED_X \
    --seed_y $SEED_Y \
    --seed_z $SEED_Z \
    --seed_radius $SEED_RADIUS \
    --chunk_size $CHUNK_SIZE \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mask_file "$MASK_FILE"

echo "Job Finished"
