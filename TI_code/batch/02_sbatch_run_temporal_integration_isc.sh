#!/bin/bash
#
usage() {
  cat <<EOF
USAGE:
  sbatch 02_sbatch_temporal_integration_isc.sh \\
         <param_file> \\
         <code_dir> \\
         <p_threshold> \\
         <data_dir> \\
         <output_dir> \\
         <mem_gb>

DESCRIPTION:
  Submits one ISC job per line in <param_file> to Sherlock via SLURM.

  Each line in <param_file> must be a single comma-separated entry:
    condition,isc_method,stats_method,n_perms
    see example isc_params_example.txt

  Example line:
    TI1_orig,loo,phaseshift,1000

ARGUMENTS:
  param_file   Text file with one ISC run per line
  code_dir     Path to TI_code/isc directory
  p_threshold  P-value threshold (e.g. 0.05)
  data_dir     Directory containing condition folders
  output_dir   Directory for ISC outputs and logs
  mem_gb       Memory per job in GB (e.g. 64)

EXAMPLE:
  sbatch 02_sbatch_temporal_integration_isc.sh \\
         isc_params_example.txt \\
         /oak/.../TI_code/isc \\
         0.05 \\
         /oak/.../hpf \\
         /oak/.../isc_analysis \\
         64
NOTES:
  - Calls 01_run_temporal_integration_isc.sh internally
  - One SLURM job is submitted per line in <param_file>
  - Designed for whole-brain ISC with high memory usage

EOF
}

if [[ $# -ne 6 ]]; then
  usage
  exit 1
fi

params_txt_file=$1
code_dir=$2 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/code/TI_code/isc
p_threshold=$3
data_dir=$4 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf
output_dir=$5 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf/isc_analysis_1000_permutations
mem=$6

cd /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/batch 

for lines in `cat $params_txt_file`; do
    line=`echo $lines | tr ',' ' '`;
    condition=`echo $line | cut -d' ' -f1`;
    isc_method=`echo $line | cut -d' ' -f2`;
    stats_method=`echo $line | cut -d' ' -f3`;
    n_perms=`echo $line | cut -d' ' -f4`;
    params=`echo "--condition ${condition} --isc_method ${isc_method} --stats_method ${stats_method} --n_perms ${n_perms}"`
    echo "bash 01_run_temporal_integration_isc.sh ${code_dir} ${params} ${p_threshold} ${data_dir} ${output_dir}"
    echo '#!/bin/bash' > isc.sbatch;
    echo "bash 01_run_temporal_integration_isc.sh ${code_dir} ${params} ${p_threshold} ${data_dir} ${output_dir}" >> isc.sbatch;
    sbatch -p owners,menon -c 16 --mem=${mem}G -o ${output_dir}/temporal_integration_log.txt isc.sbatch;
    rm TI_isc.sbatch;
    break
done