#!/bin/bash

params_txt_file=$1
code_dir=$2 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/code/TI_code/isfc
data_dir=$3 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf
output_dir=$4 #/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf/isfc_analysis_1000_permutations_hpc
mem=$5


usage() {
    echo "Usage:"
    echo " - Designed for whole-brain ISFC with high memory usage"
    echo "  sbatch 02_sbatch_temporal_integration_isfc.sh \\"
    echo "         <param_file> <code_dir> <data_dir> <output_dir> <mem_gb>"
    echo ""
    echo "Arguments:"
    echo "  <param_file>   Text file with one ISFC run per line"
    echo "                 Format:"
    echo "                   condition,isfc_method,stats_method,seed_x,seed_y,seed_z,seed_radius,n_perms,p_threshold,use_tfce"
    echo "                 Example:"
    echo "                   TI1_orig,loo,phaseshift,0,-53,2,5,1000,0.05,use_tfce"
    echo "                 Notes:"
    echo "                   use_tfce is optional. If absent, TFCE is not used."
    echo "  <code_dir>     Path to TI_code/isfc directory"
    echo "  <data_dir>     Directory containing condition folders"
    echo "  <output_dir>   Output directory for ISFC results and logs"
    echo "  <mem_gb>       Memory per job in GB (e.g. 64)"
    echo ""
    echo "Example:"
    echo " bash 02_sbatch_temporal_integration_isfc.sh \\"
    echo "   isfc_params_example.txt \\"
    echo "   /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isfc \\"
    echo "   /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf \\"
    echo "   /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/06-2025/td/hpf/isfc_analysis_1000_permutations_hpc \\"
    echo "   64"
    exit 1
}


if [ "$#" -ne 5 ]; then
    usage
fi

cd /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/batch

for lines in `cat $params_txt_file`; do
    line=`echo $lines | tr ',' ' '`;

    condition=`echo $line | cut -d' ' -f1`;
    isfc_method=`echo $line | cut -d' ' -f2`;
    stats_method=`echo $line | cut -d' ' -f3`;

    seed_x=`echo $line | cut -d' ' -f4`;
    seed_y=`echo $line | cut -d' ' -f5`;
    seed_z=`echo $line | cut -d' ' -f6`;
    seed_radius=`echo $line | cut -d' ' -f7`;
    n_perms=`echo $line | cut -d' ' -f8`;
    p_from_file=`echo $line | cut -d' ' -f9`;
    tfce_flag=`echo $line | cut -d' ' -f10`;

    params=`echo "--condition ${condition} --isfc_method ${isfc_method} --stats_method ${stats_method} --seed_x ${seed_x} --seed_y ${seed_y} --seed_z ${seed_z} --seed_radius ${seed_radius} --n_perms ${n_perms} --p_threshold ${p_from_file}"`

    if [ "$tfce_flag" = "use_tfce" ]; then
        params="${params} --use_tfce"
    fi

    echo '#!/bin/bash' > TI_isfc.sbatch;
    echo "bash 01_run_temporal_integration_isfc.sh \
    \"${code_dir}\" \
    \"${params}\" \
    \"${data_dir}\" \
    \"${output_dir}_${p_from_file}\" \
    \"${p_from_file}\"" >> TI_isfc.sbatch
    sbatch -p menon -c 16 --mem=${mem}G -t 10:00:00 -o "${output_dir}/temporal_integration_${condition}_seed_${seed_x}_${seed_y}_${seed_z}_${p_from_file}_${tfce_flag}_log.txt" TI_isfc.sbatch;
    rm TI_isfc.sbatch;
done