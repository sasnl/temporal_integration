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

cd /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/batch 



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
    output_dir_thisrun="$output_dir"


	if [ "$tfce_flag" = "use_tfce" ]; then
		params="${params} --use_tfce"
	fi

	if [ "${isc_method}" = "loo" ]; then
		params="${params} --checkpoint_every 25"
		output_dir_thisrun="${output_dir}_loo"
	fi

	if [ "${isc_method}" = "loo" ] && [ "${tfce_flag}" = "use_tfce" ]; then
		output_dir_thisrun="${output_dir}_loo_tfce"
	fi

	if [ "${isc_method}" = "pairwise" ]; then
		params="${params} --checkpoint_every 25"
		output_dir_thisrun="${output_dir}_pairwise"
	fi

	if [ "${isc_method}" = "pairwise" ] && [ "${tfce_flag}" = "use_tfce" ]; then
		output_dir_thisrun="${output_dir}_pairwise_tfce"
	fi

	if [ "${isc_method}" = "loo" ] && [ "${RESUME:-0}" -eq 1 ]; then
		params="${params} --resume"
	fi

	if [ "${isc_method}" = "pairwise" ] && [ "${RESUME:-0}" -eq 1 ]; then
		params="${params} --resume"
	fi


	echo $isc_method;
	echo $output_dir_thisrun;

    echo '#!/bin/bash' > TI_isc.sbatch;
	echo 'echo "SLURM_JOB_ID=$SLURM_JOB_ID"' >> TI_isc.sbatch
	echo 'echo "SLURM_NODELIST=$SLURM_NODELIST"' >> TI_isc.sbatch
	echo 'echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"' >> TI_isc.sbatch
	echo 'echo "SLURM_JOB_CPUS_PER_NODE=$SLURM_JOB_CPUS_PER_NODE"' >> TI_isc.sbatch
	echo 'echo "SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"' >> TI_isc.sbatch
	echo 'echo "SLURM_MEM_PER_NODE=$SLURM_MEM_PER_NODE"' >> TI_isc.sbatch
	echo 'echo "SLURM_MEM_PER_CPU=$SLURM_MEM_PER_CPU"' >> TI_isc.sbatch
	# ---- Thread control (export) ----

	echo 'export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}' >> TI_isc.sbatch
	echo 'export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}' >> TI_isc.sbatch
	echo 'export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}' >> TI_isc.sbatch
	echo 'export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}' >> TI_isc.sbatch
    
	echo "bash 01_run_temporal_integration_isc_dist.sh \
    \"${code_dir}\" \
    \"${params}\" \
    \"${data_dir}\" \
    \"${output_dir_thisrun}\" \
    \"${p_from_file}\"" >> TI_isc.sbatch
    sbatch -p menon,owners,normal -c 2 --mem=${mem}G -t 12:00:00 -o "${output_dir_thisrun}/temporal_integration_${condition}_${p_from_file}_${tfce_flag}_%j.log" TI_isc.sbatch;
	cp TI_isc.sbatch "${output_dir_thisrun}/TI_isc_${condition}_${p_from_file}_${tfce_flag}.sbatch"
done
