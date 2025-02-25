import argparse
from enum import Enum, member
from typing import Any

from .function import generate_linext, generate_weak_order, generate_weak_order_ext


class ScriptEnum(Enum):
    @member
    def LINEXT(self, *args: Any, **kwargs: Any):
        return generate_linext(*args, **kwargs)

    @member
    def WEAK_ORDER(self, *args: Any, **kwargs: Any):
        return generate_weak_order(*args, **kwargs)

    @member
    def WEAK_ORDER_EXT(self, *args: Any, **kwargs: Any):
        return generate_weak_order_ext(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.value(self, *args, **kwargs)


parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(dest="script", required=True, help="Julia script")

parser_linext = subparsers.add_parser(
    ScriptEnum.LINEXT.name, help="Generate linear extension"
)
parser_linext.add_argument("m", type=int, help="Number of criteria")
parser_linext.add_argument("-s", "--seed", type=int, help="Random seed")

parser_weak_order = subparsers.add_parser(
    ScriptEnum.WEAK_ORDER.name, help="Generate weak order"
)
parser_weak_order.add_argument("m", type=int, help="Number of criteria")
parser_weak_order.add_argument("-s", "--seed", type=int, help="Random seed")

parser_weak_order_ext = subparsers.add_parser(
    ScriptEnum.WEAK_ORDER_EXT.name, help="Generate weak order extension"
)
parser_weak_order_ext.add_argument("m", type=int, help="Number of criteria")
parser_weak_order_ext.add_argument("-s", "--seed", type=int, help="Random seed")


def parse_args():
    return parser.parse_args()
