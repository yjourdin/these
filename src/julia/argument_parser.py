import argparse

from ..enum_base import StrEnum


class ScriptEnum(StrEnum):
    LINEXT = "linext"
    WEAK_ORDER = "weak_order"
    WEAK_ORDER_EXT = "weak_order_ext"


parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest="script", required=True, help="Julia script")

parser_linext = subparsers.add_parser(
    ScriptEnum.LINEXT, help="Generate linear extension"
)
parser_linext.add_argument("m", type=int, help="Number of criteria")
parser_linext.add_argument("-s", "--seed", type=int, help="Random seed")

parser_weak_order = subparsers.add_parser(
    ScriptEnum.WEAK_ORDER, help="Generate weak order"
)
parser_weak_order.add_argument("m", type=int, help="Number of criteria")
parser_weak_order.add_argument("-s", "--seed", type=int, help="Random seed")

parser_weak_order_ext = subparsers.add_parser(
    ScriptEnum.WEAK_ORDER_EXT, help="Generate weak order extension"
)
parser_weak_order_ext.add_argument("m", type=int, help="Number of criteria")
parser_weak_order_ext.add_argument("-s", "--seed", type=int, help="Random seed")


def parse_args():
    return parser.parse_args()
