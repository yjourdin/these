import argparse
from sys import stdout

parser = argparse.ArgumentParser()
parser.add_argument("model", type=argparse.FileType("r"), help="Preferences model")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("n", type=int, help="Number of comparisons")
parser.add_argument("-e", "--error", type=float, help="Error rate")
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
