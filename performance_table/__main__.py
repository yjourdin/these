import argparse
import sys

from numpy.random import default_rng

from .generate import random_alternatives

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, help="Number of alternatives")
parser.add_argument("m", type=int, help="Number of criteria")
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument(
    "-o",
    "--output",
    default=sys.stdout,
    type=argparse.FileType("w"),
    help="Output file",
)

args = parser.parse_args()

A = random_alternatives(args.n, args.m, default_rng(args.seed))

A.data.to_csv(args.output, header=False, index=False)
