import argparse

from .arguments import Arguments


def key2int(dct):
    """Transform dict keys to int if possible

    :param dct: Dict to modify
    :return : Modified dict
    """
    try:
        return {int(k): v for k, v in dct.items()}
    except ValueError:
        return dct


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "args",
    type=argparse.FileType("r"),
    help="Arguments file",
)
parser.add_argument("-n", "--name", type=str, help="Name of the experiment")
parser.add_argument(
    "-j", "--jobs", type=int, help="Maximum number of parallel jobs"
)
parser.add_argument(
    "-s", "--seed", type=int, help="Random seed"
)


def parse_args():
    # args = parser.parse_args()
    # vars(args).update(load(args.args, object_hook=key2int))
    args = parser.parse_args()
    arguments = Arguments.from_json(args.args.read())
    arguments.name = args.name or arguments.name
    arguments.jobs = args.jobs or arguments.jobs
    arguments.seed = args.seed or arguments.seed
    return arguments
