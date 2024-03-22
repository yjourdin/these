from numpy.random import default_rng

from .argument_parser import parse_args
from .generate import random_alternatives

args = parse_args()

A = random_alternatives(args.n, args.m, default_rng(args.seed))

A.data.to_csv(args.output, header=False, index=False)
