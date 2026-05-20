from functools import reduce

from ..utils import file_or_stdout
from .args import ARGS
from .model import RMPModel, RMPParamFlag, rmp_group_model

# Create model
if ARGS.group_size == 1:
    model = RMPModel.random(
        nb_profiles=ARGS.k,
        nb_crit=ARGS.m,
        rng=ARGS.seed,
        profiles_values=ARGS.profiles_values,
    )
else:
    model_class = rmp_group_model(
        reduce(lambda x, y: x | y, ARGS.shared, RMPParamFlag.NONE)
    )
    model = model_class.random(
        group_size=ARGS.group_size,
        nb_profiles=ARGS.k,
        nb_crit=ARGS.m,
        rng=ARGS.seed,
        profiles_values=ARGS.profiles_values,
    )


# Write output
with file_or_stdout(ARGS.output, "w") as f:
    f.write(model.to_json())
