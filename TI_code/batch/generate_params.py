import datetime

def _timestamp():
    # Example: 20260122_134455, to append to files generated
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_isc_params():
    conditions = ['TI1_orig', 'TI1_sent', 'TI1_word']
    methods = ['loo', 'pairwise']
    tfces = [False, True]


    ts = _timestamp()
    out_base = f"./isc_params_{ts}"
    out_no_tfce = f"{out_base}_no_tfce.txt"
    out_tfce = f"{out_base}_tfce.txt"

    with open(out_no_tfce, "w") as f_no, open(out_tfce, "w") as f_tfce:
        for cond in conditions:
            for method in methods:
                for use_tfce in tfces:
                    # cmd_args = f"--condition {cond} --isc_method {method} --stats_method phaseshift --n_perms 1000 --p_threshold 0.05"
                    cmd_args = f"{cond},{method},phaseshift,1000,0.05"

                    if use_tfce:
                        f_tfce.write(f"{cmd_args},use_tfce\n")
                    else:
                        f_no.write(f"{cmd_args}\n")
    return out_no_tfce, out_tfce


def generate_isfc_params():
    conditions = ['TI1_orig', 'TI1_sent', 'TI1_word']
    methods = ['loo', 'pairwise']
    tfces = [False, True]
    
    seeds = [
        ("PMC", 0, -53, 2),
        ("L-pSTS", -63, -42, 9),
        ("R-pSTS", 57, -31, 5)
    ]
    ts = _timestamp()
    out_base = f"./isfc_params_{ts}"
    out_no_tfce = f"{out_base}_no_tfce.txt"
    out_tfce = f"{out_base}_tfce.txt"

    with open(out_no_tfce, "w") as f_no, open(out_tfce, "w") as f_tfce:
        for seed_name, x, y, z in seeds:
            for cond in conditions:
                for method in methods:
                    for use_tfce in tfces:
                        # cmd_args = f"--condition {cond} --isfc_method {method} --stats_method phaseshift --seed_x {x} --seed_y {y} --seed_z {z} --seed_radius 5 --n_perms 1000 --p_threshold 0.05"
                        # cmd_args = f"--condition {cond} --isfc_method {method} --stats_method phaseshift --seed_x {x} --seed_y {y} --seed_z {z} --seed_radius 5 --n_perms 1000 --p_threshold 0.05"
                        cmd_args = f"{cond},{method},phaseshift,{x},{y},{z},5,1000,0.05"
                        if use_tfce:
                            f_tfce.write(f"{cmd_args},use_tfce\n")
                        else:
                            f_no.write(f"{cmd_args}\n")
    return out_no_tfce, out_tfce


if __name__ == "__main__":
    isc_no_tfce, isc_tfce = generate_isc_params()
    isfc_no_tfce, isfc_tfce = generate_isfc_params()

    print("Generated:")
    print(f"  {isc_no_tfce}")
    print(f"  {isc_tfce}")
    print(f"  {isfc_no_tfce}")
    print(f"  {isfc_tfce}")
