
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
64;

bash 02_sbatch_run_temporal_integration_isc.sh \
isc_params_20260122_133305_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc \
/scratch/users/daelsaid/updated_results/td/ \
/scratch/users/daelsaid/updated_results/td/isc_analysis_1000_permutations_hpc \
64;


bash 02_sbatch_run_temporal_integration_isfc.sh \
isfc_params_20260122_133305_no_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isfc \
/scratch/users/daelsaid/updated_results/td/ \
/scratch/users/daelsaid/updated_results/td/isfc_analysis_1000_permutations_hpc \
64;

bash 02_sbatch_run_temporal_integration_isc.sh \
isfc_params_20260122_133305_tfce.txt \
/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isfc \
/scratch/users/daelsaid/updated_results/td/ \
/scratch/users/daelsaid/updated_results/td/isfc_analysis_1000_permutations_hpc \
64;
