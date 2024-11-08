import argparse

from ..enum_base import StrEnum


class TypeEnum(StrEnum):
    PREFERENCE_STRUCTURE = "PS"
    RANKING = "R"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=argparse.FileType("r"), help="Preferences model")
parser.add_argument("A", type=argparse.FileType("r"), help="Alternatives")

subparsers = parser.add_subparsers(dest="type", required=True, help="Output type")

parser_PS = subparsers.add_parser(
    TypeEnum.PREFERENCE_STRUCTURE, help="Preference structure"
)
parser_PS.add_argument("-n", type=int, help="Number of comparisons")
parser_PS.add_argument("-e", "--error", type=float, help="Error rate")
parser_PS.add_argument("--same", action="store_true", help="Same alternatives")
parser_PS.add_argument("-s", "--seed", type=int, help="Random seed")
parser_PS.add_argument("--seed-shuffle", type=int, help="Shuffle random seed")
parser_PS.add_argument("--seed-error", type=int, help="Error random seed")

parser_R = subparsers.add_parser(TypeEnum.RANKING, help="Ranking")

parser.add_argument(
    "-o",
    "--output",
    default="stdout",
    help="Output filename",
)


def parse_args():
    return parser.parse_args()
