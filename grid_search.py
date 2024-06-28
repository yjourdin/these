from json import dump, load

from src.main.config import SAConfig

configs = []

MAX_ITER = 20_000

with open("args.json", "r+") as f:
    args = load(f)
    f.seek(0)

    for accept in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for alpha in [0.9, 0.95, 0.99, 0.995, 0.999]:
            for amp in [0.1, 0.2, 0.3, 0.4, 0.5] if "SRMP" in args["Me"] else [0]:
                configs.append(SAConfig(accept, alpha, amp, MAX_ITER).to_dict())

    args["config"] = configs

    dump(args, f, indent=4)
    f.truncate()
