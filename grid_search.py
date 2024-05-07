from json import dump, load

from run.config import SAConfig

configs = {}

MAX_ITER = 20_000

id = 0
for T0_coef in [0.1, 0.5, 1, 5, 10]:
    for alpha in [0.999, 0.9995, 0.9999, 0.99995, 0.99999]:
        for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
            configs[id] = SAConfig(T0_coef, alpha, amp, MAX_ITER).to_dict()
            id += 1

with open("args.json", "r+") as f:
    args = load(f)
    f.seek(0)

    args["config"]["SA"] = configs

    dump(args, f, indent=4)
    f.truncate()
