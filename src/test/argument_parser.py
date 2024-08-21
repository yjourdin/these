import argparse

parser = argparse.ArgumentParser()
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("Mo", type=argparse.FileType("r"), help="Original model")
parser.add_argument("Me", type=argparse.FileType("r"), help="Elicited model")
parser.add_argument("-r", "--result", type=argparse.FileType("a"), help="Result file")


def parse_args():
    return parser.parse_args()
