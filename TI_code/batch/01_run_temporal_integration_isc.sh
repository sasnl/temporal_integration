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

usage() {
    echo ""
    echo "USAGE:"
    echo "  $0 <code_path> <param_string> <p_threshold> <data_dir> <output_dir>"
    echo ""
    echo "ARGUMENTS:"
    echo "  code_path Absolute path to the ISC code directory containing run_isc_pipeline.py"\"
    echo "      Example:\"/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc"\"
    echo ""
    echo "  param_string: Quoted parameter string passed directly to run_isc_pipeline.py"
    echo "      This should match the argparse flags expected by the pipeline."
    echo "      Example: \"--condition TI1_orig --isc_method loo --stats_method bootstrap --n_perms 1000\""
    echo ""
    echo "  data_dir"
    echo "      Path to preprocessed fMRI data directory containing condition subfolders"
    echo "      Example:"
    echo "        /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/"
    echo "        post_processed_wholebrain/filtered/06-2025/td/hpf"
    echo ""
    echo "  output_dir: Output directory where ISC results, maps, and logs will be written"
    echo "      Example: /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td"
    echo ""
    echo "  p_threshold: P value threshold used for statistical masking"\"
    echo "      Example: 0.05"\"
    echo ""

    echo "EXAMPLE COMMAND:"
    echo "  $0 \\"
    echo "    /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/"
    echo "    taskfmri/temporal_integration/TI_code/isc \\"
    echo "    \"--condition TI1_orig --isc_method loo --stats_method bootstrap --n_perms 1000\" \\"
    echo "    0.05 \\"
    echo "    /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf \\"
    echo "    /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf/isc_analysis_1000_permutations_0.05"
    echo ""
    exit 1
}

# ---- argument check ----
if [ "$#" -ne 5 ]; then
    echo "ERROR: Incorrect number of arguments."
    usage
fi


code_path=$1 #" /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc
param_string=$2 #row from isc_params_example.txt
#"TI1_orig, TI1_sent, TI1_"
data_dir=$3
output_dir=$4
thresh=$5

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
${env_path}/bin/python ${code_path}/run_isc_pipeline.py \
${param_string} \
 "--data_dir" ${data_dir} \
 "--output_dir" ${output_dir}_${thresh} \
 "--mask_file" ${mask_file}
 
#   param_string="--condition TI1_sent --isc_method loo --stats_method phaseshift --n_perms 1000";p_threshold=0.05; data_dir=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf;  output_dir=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf/isc_analysis_1000_permutations_hpc;env_path=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/isc_env;mask_file="/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/mask/MNI152_T1_2mm_brain_mask.nii";
