import argparse

from ..jobs import JOBS
from .arguments import Arguments

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "args",
    type=argparse.FileType("r"),
    help="Arguments file",
)
parser.add_argument("-j", "--jobs", default=JOBS, type=int, help="Number of jobs")


def parse_args():
    args = parser.parse_args()
    with args.args as file:
        arguments = Arguments.from_json(file.read())
    arguments.jobs = args.jobs
    return arguments
