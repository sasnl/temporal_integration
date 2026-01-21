
def generate_isc_params():
    conditions = ['TI1_orig', 'TI1_sent', 'TI1_word']
    methods = ['loo', 'pairwise']
    tfces = [False, True]
    
    with open('batch/isc_params.txt', 'w') as f:
        for cond in conditions:
            for method in methods:
                for use_tfce in tfces:
                    cmd_args = f"--condition {cond} --isc_method {method} --stats_method phaseshift --n_perms 1000 --p_threshold 0.05"
                    if use_tfce:
                        cmd_args += " --use_tfce"
                    f.write(f"{cmd_args}\n")

def generate_isfc_params():
    conditions = ['TI1_orig', 'TI1_sent', 'TI1_word']
    methods = ['loo', 'pairwise']
    tfces = [False, True]
    
    seeds = [
        ("PMC", 0, -53, 2),
        ("L-pSTS", -63, -42, 9),
        ("R-pSTS", 57, -31, 5)
    ]
    
    with open('batch/isfc_params.txt', 'w') as f:
        for seed_name, x, y, z in seeds:
            for cond in conditions:
                for method in methods:
                    for use_tfce in tfces:
                        cmd_args = f"--condition {cond} --isfc_method {method} --stats_method phaseshift --seed_x {x} --seed_y {y} --seed_z {z} --seed_radius 5 --n_perms 1000 --p_threshold 0.05"
                        if use_tfce:
                            cmd_args += " --use_tfce"
                        f.write(f"{cmd_args}\n")

if __name__ == "__main__":
    generate_isc_params()
    generate_isfc_params()
    print("Generated batch/isc_params.txt and batch/isfc_params.txt")
