from json import dump, load

from src.constants import DEFAULT_MAX_TIME
from src.main.experiments.elicitation.config import SAConfig

ARGS_FILE = "args_elicitation.json"

configs = []

with open(ARGS_FILE, "r+") as f:
    args = load(f)
    f.seek(0)

    for accept in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for alpha in [0.9, 0.99, 0.999, 0.9999, 0.99999]:
            configs.append(
                SAConfig.json_to_dict(
                    SAConfig(DEFAULT_MAX_TIME, accept, alpha).to_json()
                )
            )

    args["config"] = configs

    dump(args, f, indent=4)
    f.truncate()
