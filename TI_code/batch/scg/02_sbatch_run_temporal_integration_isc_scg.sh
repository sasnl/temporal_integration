#!/bin/bash

params_txt_file=$1
code_dir=$2 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/code/TI_code/isc
data_dir=$3 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf
output_dir=$4 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf/isc_analysis_1000_permutations_hpc
mem=$5


usage() {
    echo "Usage:"
    echo " - Designed for whole-brain ISC with high memory usage"
    echo "  sbatch 02_sbatch_temporal_integration_isc.sh \\"
    echo "         <param_file> <code_dir> <data_dir> <output_dir> <mem_gb>"
    echo ""
    echo "Arguments:"
    echo "  <param_file>   Text file with one ISC run per line"
    echo "                 Format: condition,isc_method,stats_method,n_perms"
    echo "  <code_dir>     Path to TI_code/isc directory"
    echo "  <data_dir>     Directory containing condition folders"
    echo "  <output_dir>   Output directory for ISC results and logs"
    echo "  <mem_gb>       Memory per job in GB (e.g. 64)"
    echo ""
    echo "Example:"
    echo " bash 02_sbatch_run_temporal_integration_isc.sh isc_params_example.txt /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/ /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/isc_analysis_1000_permutations_hpc_0.05 32;"
    exit 1 
}

if [ "$#" -ne 5 ]; then
    usage
fi

cd /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/batch/scg

for lines in `cat $params_txt_file`; do
    line=`echo $lines | tr ',' ' '`;
    condition=`echo $line | cut -d' ' -f1`;
    isc_method=`echo $line | cut -d' ' -f2`;
    stats_method=`echo $line | cut -d' ' -f3`;
    n_perms=`echo $line | cut -d' ' -f4`;
    p_from_file=`echo $line | cut -d' ' -f5`
    tfce_flag=`echo $line | cut -d' ' -f6`


    # if [ -n "$p_from_file" ]; then
    #     pval="$p_from_file"
    # else
    #     pval="$p_threshold"
    # fi
    
    params=`echo "--condition ${condition} --isc_method ${isc_method} --stats_method ${stats_method} --n_perms ${n_perms} --p_threshold ${p_from_file}"`

    if [ "$tfce_flag" = "use_tfce" ]; then
        params="${params} --use_tfce"
        output_dir_thisrun="${output_dir}_thresholded"
    fi

	if [ "${stats_method}" = "phaseshift" ]; then
		params="${params} --checkpoint_every 25"
	fi

	if [ "${stats_method}" = "phaseshift" ] && [ "${RESUME:-0}" -eq 1 ]; then
		params="${params} --resume"
	fi

    output_dir_thisrun="$output_dir"

    echo '#!/bin/bash' > TI_isc_scg.sbatch;
	echo 'echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"' >> TI_isc_scg.sbatch
	echo 'export OMP_NUM_THREADS=1' >> TI_isc_scg.sbatch
	echo 'export OPENBLAS_NUM_THREADS=1' >> TI_isc_scg.sbatch
	echo 'export MKL_NUM_THREADS=1' >> TI_isc_scg.sbatch
	echo 'export NUMEXPR_NUM_THREADS=1' >> TI_isc_scg.sbatch
    echo "bash 01_run_temporal_integration_isc_scg.sh \
    \"${code_dir}\" \
    \"${params}\" \
    \"${data_dir}\" \
    \"${output_dir_thisrun}\" \
    \"${p_from_file}\"" >> TI_isc_scg.sbatch
	sbatch --job-name=TI_isc_scg --nodes=1 --ntasks=1 --cpus-per-task=4 --partition=nih_s10 --account=menon --time=12:00:00 --mem=${mem} -o "${output_dir_thisrun}/temporal_integration_${condition}_${p_from_file}_${tfce_flag}_scg_%j.log" TI_isc_scg.sbatch; 
    rm TI_isc_scg.sbatch;
done
