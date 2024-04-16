from json import dump, load

configs = []

N_bc = list(range(100, 1100, 100)) + [2000]

id = 0
for T0_coef in [0.1, 0.5, 1, 5, 10]:
    for Tf_coef in [0.1, 0.5, 1, 5, 10]:
        for alpha in [0.9]:
            for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
                configs.append(
                    {
                        "id": id,
                        "T0": {str(n): T0_coef * 1 / n for n in N_bc},
                        "Tf": {str(n): Tf_coef * 1 / (10 * n) for n in N_bc},
                        "alpha": alpha,
                        "amp": amp,
                    }
                )
                id += 1

with open("args.json", "r+") as f:
    args = load(f)
    f.seek(0)

    args["config"] = configs

    dump(args, f, indent=4)
    f.truncate()
