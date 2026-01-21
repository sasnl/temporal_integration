loop used to run multiple thresholds:


for thresh in 0.05 0.005 0.01 0.001; do 
    bash 02_sbatch_run_temporal_integration_isc.sh isc_params_example.txt /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/scripts/taskfmri/temporal_integration/TI_code/isc ${thresh} /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/ /oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/filtered/updated_results/td/isc_analysis_1000_permutations_hpc_${thresh} 32;
done