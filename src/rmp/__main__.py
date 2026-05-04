from .args import ARGS
from .model import RMPModel, rmp_group_model

# Create model
if ARGS.group_size == 1:
    model = RMPModel.random(
        nb_profiles=ARGS.k,
        nb_crit=ARGS.m,
        rng=ARGS.seed,
        profiles_values=ARGS.profiles_values,
    )
else:
    model_class = rmp_group_model(ARGS.shared)
    model = model_class.random(
        group_size=ARGS.group_size,
        nb_profiles=ARGS.k,
        nb_crit=ARGS.m,
        rng=ARGS.seed,
        profiles_values=ARGS.profiles_values,
    )


# Write results
ARGS.output.write(model.to_json())
