import argparse
from dataclasses import dataclass, field
from pathlib import Path

from ..dataclass import Dataclass
from .model import RMPParamFlag

parser = argparse.ArgumentParser()
parser.add_argument("group_size", type=int, default=1, help="Group size")
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("m", type=int, help="Number of criteria")
parser.add_argument(
    "--shared",
    nargs="*",
    default=[],
    type=RMPParamFlag,
    choices=RMPParamFlag,
    help="Parameters shared between decision makers",
)
parser.add_argument(
    "-p", "--profiles-values", type=Path, help="Possible values for profiles"
)
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument("-o", "--output", type=Path, help="Output file")


@dataclass(init=False)
class Arguments(Dataclass):
    group_size: int
    k: int
    m: int
    shared: list[RMPParamFlag] = field(default_factory=list)
    profiles_values: Path | None = None
    seed: int | None = None
    output: Path | None = None


ARGS = parser.parse_args(namespace=Arguments())
