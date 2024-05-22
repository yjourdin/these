from numpy.random import default_rng

from .argument_parser import parse_args
from .generate import random_alternatives

# Parse arguments
args = parse_args()


# Create performance table
A = random_alternatives(args.n, args.m, default_rng(args.seed))


# Write results
A.data.to_csv(args.output, header=False, index=False)
