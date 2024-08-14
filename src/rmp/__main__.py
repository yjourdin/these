from numpy.random import default_rng

from .argument_parser import parse_args
from .model import RMPModel

# Parse arguments
args = parse_args()


# Create model
if args.balanced:
    model = RMPModel.balanced(
        args.k, args.m, default_rng(args.seed), args.profiles_values
    )
else:
    model = RMPModel.random(
        args.k, args.m, default_rng(args.seed), args.profiles_values
    )


# Write results
args.output.write(model.to_json())
