import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", type=argparse.FileType("r"), help="Collective model")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")
parser.add_argument("D", nargs="+", type=argparse.FileType("r"), help="Comparisons")
parser.add_argument("-o", "--output", help="Output filename")


def parse_args():
    return parser.parse_args()
