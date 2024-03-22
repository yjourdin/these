from numpy.random import default_rng

from .argument_parser import parse_args
from .generate import balanced_srmp, random_srmp

args = parse_args()

if args.balanced:
    model = balanced_srmp(args.k, args.m, default_rng(args.seed), args.profiles_values)
else:
    model = random_srmp(args.k, args.m, default_rng(args.seed), args.profiles_values)

args.output.write(model.to_json())
