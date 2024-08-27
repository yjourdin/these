import argparse

from .arguments import Arguments

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("args", type=argparse.FileType("r"), help="Arguments file")
parser.add_argument("-d", "--dir", type=str, help="Results directory")
parser.add_argument("-n", "--name", type=str, help="Experiment name")
parser.add_argument("-j", "--jobs", type=int, help="Number of jobs")
parser.add_argument("-s", "--stop-error", action="store_true", help="Stop on error")


def parse_args():
    args = parser.parse_args()
    with args.args as file:
        arguments = Arguments.from_json(file.read())
    arguments.dir = args.dir or arguments.dir
    arguments.name = args.name or arguments.name
    arguments.jobs = args.jobs or arguments.jobs
    arguments.stop_error = args.stop_error
    return arguments
