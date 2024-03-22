import argparse
from sys import stdout

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, help="Number of alternatives")
parser.add_argument("m", type=int, help="Number of criteria")
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument(
    "-o",
    "--output",
    default=stdout,
    type=argparse.FileType("w"),
    help="Output file",
)


def parse_args():
    return parser.parse_args()
