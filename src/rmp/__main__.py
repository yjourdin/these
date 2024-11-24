from ..random import rng
from .argument_parser import parse_args
from .model import RMPModel, rmp_group_model

# Parse arguments
args = parse_args()


# Create model
if args.group_size == 1:
    model = RMPModel.random(
        nb_profiles=args.k,
        nb_crit=args.m,
        rng=rng(args.seed),
        profiles_values=args.profiles_values,
    )
else:
    model_class = rmp_group_model(args.shared)
    model = model_class.random(
        group_size=args.group_size,
        nb_profiles=args.k,
        nb_crit=args.m,
        rng=rng(args.seed),
        profiles_values=args.profiles_values,
    )


# Write results
args.output.write(model.to_json())
