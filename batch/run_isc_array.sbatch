#!/bin/bash
#SBATCH --job-name=ISC_Array
#SBATCH --output=logs/isc_%A_%a.out
#SBATCH --error=logs/isc_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=1-12
#SBATCH --partition=normal

# Load environment (adjust as needed for Sherlock)
module load python/3.9.0
source $HOME/isc_env/bin/activate

# Get arguments from the params file
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" batch/isc_params.txt)

echo "Running task ${SLURM_ARRAY_TASK_ID} with args: $ARGS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Run the pipeline
# Note: Adjust paths if necessary. Assuming submission from code/TI_code
python isc/run_isc_pipeline.py $ARGS
