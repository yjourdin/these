from json import dump, load

from src.constants import DEFAULT_MAX_TIME
from src.main.experiments.elicitation.config import SAConfig, SRMPSAConfig

configs = []

MAX_ITER = 20_000

with open("args.json", "r+") as f:
    args = load(f)
    f.seek(0)

    for accept in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for alpha in [0.9, 0.95, 0.99, 0.995, 0.999]:
            if "SRMP" in args["Me"]:
                for amp in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    configs.append(
                        SRMPSAConfig(
                            DEFAULT_MAX_TIME, accept, alpha, MAX_ITER, amp
                        ).to_dict()
                    )
            else:
                configs.append(
                    SAConfig(DEFAULT_MAX_TIME, accept, alpha, MAX_ITER).to_dict()
                )

    args["config"] = configs

    dump(args, f, indent=4)
    f.truncate()
