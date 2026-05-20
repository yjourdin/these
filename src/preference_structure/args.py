import argparse
from dataclasses import dataclass
from pathlib import Path

from src.case_insensitive_str_enum import CaseInsensitiveStrEnum

from ..dataclass import Dataclass


class TypeEnum(CaseInsensitiveStrEnum):
    PREFERENCE_STRUCTURE = "PS"
    RANKING = "R"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path, help="Preferences model")
parser.add_argument("A", type=Path, help="Alternatives")

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

parser.add_argument("-o", "--output", type=Path, help="Output filename")


@dataclass(init=False)
class Arguments(Dataclass):
    model: Path
    A: Path
    type: TypeEnum
    n: int | None = None
    error: float | None = None
    same: bool = False
    seed: int | None = None
    seed_shuffle: int | None = None
    seed_error: int | None = None
    output: Path | None = None


ARGS = parser.parse_args(namespace=Arguments())
