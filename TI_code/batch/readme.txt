bash 02_sbatch_run_temporal_integration_isc.sh \
isc_params_20260122_133305_no_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/ \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/isc_analysis_1000_permutations_hpc \
64;

bash 02_sbatch_run_temporal_integration_isc.sh \
isc_params_20260122_133305_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/ \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/isc_analysis_1000_permutations_hpc \
64;

bash 02_sbatch_run_temporal_integration_isfc.sh \
isfc_params_20260122_133305_no_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isfc \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/ \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/isfc_analysis_1000_permutations_hpc \
64;

bash 02_sbatch_run_temporal_integration_isc.sh \
isfc_params_20260122_133305_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isfc \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/ \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/isfc_analysis_1000_permutations_hpc \
64;


bash 02_sbatch_run_temporal_integration_isc.sh \
isc_params_20260122_133305_no_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc \
/scratch/users/daelsaid/updated_results/td/ \
/scratch/users/daelsaid/updated_results/td/isc_analysis_1000_permutations_hpc \
96;

bash 02_sbatch_run_temporal_integration_isc.sh \
isc_params_20260122_133305_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc \
/scratch/users/daelsaid/updated_results/td/ \
/scratch/users/daelsaid/updated_results/td/isc_analysis_1000_permutations_hpc \
96;

bash 02_sbatch_run_temporal_integration_isfc.sh \
isfc_params_20260122_133305_no_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isfc \
/scratch/users/daelsaid/updated_results/td/ \
/scratch/users/daelsaid/updated_results/td/isfc_analysis_1000_permutations_hpc \
128;

bash 02_sbatch_run_temporal_integration_isfc.sh \
isfc_params_20260122_133305_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isfc \
/scratch/users/daelsaid/updated_results/td/ \
/scratch/users/daelsaid/updated_results/td/isfc_analysis_1000_permutations_hpc \
128;


# ### restart:
# env_path=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/isc_env
# source ${env_path}/bin/activate

# isc_dir=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc
# mask_file=/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/mask/MNI152_T1_2mm_brain_mask.nii

# outdir=/scratch/users/daelsaid/updated_results/td/isc_analysis_1000_permutations_hpc_thresholded_TI1_orig
# nperms=1000
# pthr=0.05
# stats_method=bootstrap

# ${env_path}/bin/python ${isc_dir}/isc_stats.py \
#   --stats_method ${stats_method} \
#   --input_map ${outdir}/isc_${condition}_loo_desc-zscore.nii.gz \
#   --output_dir ${outdir} \
#   --mask_file ${mask_file} \
#   --n_perms ${nperms} \
#   --p_threshold ${pthr} \
#   --use_tfce


# ${env_path}/bin/python ${isc_dir}/isc_stats.py \
#   --stats_method ${stats_method} \
#   --input_map ${outdir}/isc_${condition}_pairwise_desc-zscore.nii.gz \
#   --output_dir ${outdir} \
#   --mask_file ${mask_file} \
#   --n_perms ${nperms} \
#   --p_threshold ${pthr} \
#   --use_tfce
