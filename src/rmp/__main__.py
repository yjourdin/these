from numpy.random import default_rng

from .argument_parser import parse_args
from .model import RMPModel, rmp_group_model

# Parse arguments
args = parse_args()


# Create model
if args.size == 1:
    if args.balanced:
        model = RMPModel.balanced(
            nb_profiles=args.k,
            nb_crit=args.m,
            rng=default_rng(args.seed),
            profiles_values=args.profiles_values,
        )
    else:
        model = RMPModel.random(
            nb_profiles=args.k,
            nb_crit=args.m,
            rng=default_rng(args.seed),
            profiles_values=args.profiles_values,
        )
else:
    model_class = rmp_group_model(args.shared)
    if args.balanced:
        model = model_class.balanced(
            size=args.size,
            nb_profiles=args.k,
            nb_crit=args.m,
            rng=default_rng(args.seed),
            profiles_values=args.profiles_values,
        )
    else:
        model = model_class.random(
            size=args.size,
            nb_profiles=args.k,
            nb_crit=args.m,
            rng=default_rng(args.seed),
            profiles_values=args.profiles_values,
        )


# Write results
args.output.write(model.to_json())
