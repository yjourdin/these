from ..random import rng
from .argument_parser import parse_args
from .model import SRMPModel, srmp_group_model

# parse arguments
args = parse_args()

# Create model
if args.group_size == 1:
    if args.balanced:
        model = SRMPModel.balanced(
            nb_profiles=args.k,
            nb_crit=args.m,
            rng=rng(args.seed),
            profiles_values=args.profiles_values,
        )
    else:
        model = SRMPModel.random(
            nb_profiles=args.k,
            nb_crit=args.m,
            rng=rng(args.seed),
            profiles_values=args.profiles_values,
        )
else:
    model_class = srmp_group_model(args.shared)
    if args.balanced:
        model = model_class.balanced(
            group_size=args.group_size,
            nb_profiles=args.k,
            nb_crit=args.m,
            rng=rng(args.seed),
            profiles_values=args.profiles_values,
        )
    else:
        model = model_class.random(
            group_size=args.group_size,
            nb_profiles=args.k,
            nb_crit=args.m,
            rng=rng(args.seed),
            profiles_values=args.profiles_values,
        )


# Write results
args.output.write(model.to_json())
