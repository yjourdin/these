import argparse
from json import load


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
parser.add_argument("name", type=str, help="Name of the experiment")
parser.add_argument(
    "args",
    type=argparse.FileType("r"),
    help="Arguments file",
)
parser.add_argument(
    "-j", "--jobs", default=35, type=int, help="Maximum number of parallel jobs"
)


def parse_args():
    args = parser.parse_args()
    vars(args).update(load(args.args, object_hook=key2int))
    return args
