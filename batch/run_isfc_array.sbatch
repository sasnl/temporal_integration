#!/bin/bash
#SBATCH --job-name=ISFC_Array
#SBATCH --output=logs/isfc_%A_%a.out
#SBATCH --error=logs/isfc_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=1-36
#SBATCH --partition=normal

# Load environment (adjust as needed for Sherlock)
module load python/3.9.0
source $HOME/isc_env/bin/activate

# Get arguments from the params file
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" batch/isfc_params.txt)

echo "Running task ${SLURM_ARRAY_TASK_ID} with args: $ARGS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Run the pipeline
python isfc/run_isfc_pipeline.py $ARGS
