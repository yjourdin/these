import argparse
import sys

from numpy.random import default_rng

from .generate import balanced_rmp, random_rmp

parser = argparse.ArgumentParser()
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("m", type=int, help="Number of criteria")
parser.add_argument("-b", "--balanced", action="store_true", help="Balanced model")
parser.add_argument(
    "-p",
    "--profiles-values",
    type=argparse.FileType("w"),
    help="Possible values for profiles",
)
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument(
    "-o",
    "--output",
    default=sys.stdout,
    type=argparse.FileType("w"),
    help="Output file",
)

args = parser.parse_args()

if args.balanced:
    model = balanced_rmp(args.k, args.m, default_rng(args.seed), args.profiles_values)
else:
    model = random_rmp(args.k, args.m, default_rng(args.seed), args.profiles_values)

args.output.write(model.to_json())
