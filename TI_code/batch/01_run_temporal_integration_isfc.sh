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

code_path=$1 #" /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc
param_string=$2 #row from isc_params_example.txt
#"TI1_orig, TI1_sent, TI1_"
data_dir=$3
output_dir=$4
thresh=$5


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
    echo "  $0  \\"
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

env_path=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/isc_env
mask_file="/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/mask/MNI152_T1_2mm_brain_mask.nii"

source ${env_path}/bin/activate
param_string="$(echo "$param_string" | tr '\n\t' ' ')"
read -r -a PARAM_ARR <<< "$param_string"

echo "Param string received:"
echo "  $param_string"
echo "Expanded argv:"
printf "  %q\n" "${PARAM_ARR[@]}"

#run script 
${env_path}/bin/python ${code_path}/run_isfc_pipeline.py \
${param_string} \
 "--data_dir" ${data_dir} \
 "--output_dir" ${output_dir} \
 "--mask_file" ${mask_file}
 
#   param_string="--condition TI1_sent --isc_method loo --stats_method phaseshift --n_perms 1000";p_threshold=0.05; data_dir=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf;  output_dir=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf/isc_analysis_1000_permutations_hpc;env_path=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/isc_env;mask_file="/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/mask/MNI152_T1_2mm_brain_mask.nii";
#--condition TI1_orig --seed_x 45 --seed_y -30 --seed_z 10 --seed_radius 5 --stats_method phaseshift --n_perms 1000
