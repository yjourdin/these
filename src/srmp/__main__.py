from numpy.random import default_rng

from .argument_parser import parse_args
from .model import SRMPModel

# parse arguments
args = parse_args()


# Create model
if args.balanced:
    model = SRMPModel.balanced(
        args.k, args.m, default_rng(args.seed), args.profiles_values
    )
else:
    model = SRMPModel.random(
        args.k, args.m, default_rng(args.seed), args.profiles_values
    )


# Write results
args.output.write(model.to_json())
