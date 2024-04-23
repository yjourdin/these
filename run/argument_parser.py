import argparse

from .arguments import Arguments

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "args",
    type=argparse.FileType("r"),
    help="Arguments file",
)


def parse_args():
    args = parser.parse_args()
    arguments = Arguments.from_json(args.args.read())
    return arguments
