#!/bin/bash
#SBATCH --job-name=ti_isfc_array
#SBATCH --output=logs/isfc_array_%A_%a.out
#SBATCH --error=logs/isfc_array_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=normal
#SBATCH --array=1-2
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your_email@stanford.edu

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_ROOT="/scratch/users/tongshan/temporal_integration"
ENV_NAME="isc_env"

# Paths (Override defaults here)
DATA_DIR=""
OUTPUT_DIR="$PROJECT_ROOT/results"
MASK_FILE="$PROJECT_ROOT/TI_code/mask/MNI152_T1_2mm_brain_mask.nii"

# Parameter File
# Create this file with one job configuration per line
PARAM_FILE="$PROJECT_ROOT/code/TI_code/batch/isfc_params_example.txt"

# Default shared args
CHUNK_SIZE=300000
SEED_RADIUS=5

# ==============================================================================
# EXECUTION
# ==============================================================================

# 1. Load Modules
ml python/3.9.0
ml py-numpy/1.20.3_py39
ml py-scipy/1.6.3_py39
ml openmpi

# 2. Activate Virtual Environment
source $PROJECT_ROOT/code/TI_code/isc_env/bin/activate

# 3. Get Parameters for this Array Task ID
ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $PARAM_FILE)

echo "Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running arguments: $ARGS"

# 4. Run Pipeline
# Note: seed_radius is often constant, but can be overridden in ARGS if needed
# ARGS from file will take precedence if python argparse is used correctly, 
# but here we pass defaults. Ensure param file args don't conflict or are clear.
# To be safe, we pass defaults here, but specific seeds in ARGS.

python3 $PROJECT_ROOT/code/TI_code/run_isfc_pipeline.py \
    $ARGS \
    --chunk_size $CHUNK_SIZE \
    --seed_radius $SEED_RADIUS \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --mask_file "$MASK_FILE"

echo "Job Finished"
