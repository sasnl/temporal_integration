#!/bin/bash
#
# Temporal Integration — Inter-Subject Correlation (ISC) Pipeline
#
# Author: Dawlat M. El-Said
# Affiliation: Stanford University
#
# Description:
#   Batch and execution scripts for running whole-brain or ROI-based
#   Inter-Subject Correlation (ISC) analyses on Sherlock (SLURM),
#   using the Temporal Integration codebase.
#
usage() {
    echo ""
    echo "USAGE:"
    echo "  $0 <code_path> <param_string> <p_threshold> <data_dir> <output_dir>"
    echo ""
    echo "ARGUMENTS:"
    echo "  code_path     Absolute path to isc directory with directory"
    echo "  param_string  Quoted parameter string passed to run_isc_pipeline.py"
    echo "                Example:"
    echo "                  \"--condition TI1_orig --isc_method loo --stats_method bootstrap --n_perms 1000\""
    echo "  p_threshold   P-value threshold (e.g. 0.05)"
    echo "  data_dir      Path to preprocessed fMRI data"
    echo "  output_dir    Output directory for ISC results"
    echo ""
    echo "EXAMPLE:"
    echo "  $0 /oak/.../TI_code/isc \\"
    echo "     \"--condition TI1_orig --isc_method loo --stats_method bootstrap --n_perms 1000\" \\"
    echo "     0.05 \\"
    echo "     /oak/.../td/hpf \\"
    echo "     /oak/.../td/hpf/isc_analysis"
    echo ""
    exit 1
}

# ---- argument check ----
if [ "$#" -ne 5 ]; then
    echo "ERROR: Incorrect number of arguments."
    usage
fi


code_path=$1 #"oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/code/TI_code/isc" 
param_string=$2 #row from isc_params_example.txt
#"TI1_orig, TI1_sent, TI1_"
p_threshold=$3
data_dir=$4
output_dir=$5

env_path=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/isc_env
mask_file="/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/mask/MNI152_T1_2mm_brain_mask.nii"

source ${env_path}/bin/activate

#run script 
${env_path}/bin/python ${code_path}/run_isc_pipeline.py \
${param_string} \
 "--p_threshold" ${p_threshold} \
 "--data_dir" ${data_dir} \
 "--output_dir" ${output_dir} \
 "--mask_file" ${mask_file}