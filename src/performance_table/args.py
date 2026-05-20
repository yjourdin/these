import argparse
from dataclasses import dataclass
from pathlib import Path

from ..dataclass import Dataclass

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int, help="Number of alternatives")
parser.add_argument("m", type=int, help="Number of criteria")
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument("-o", "--output", type=Path, help="Output file")


@dataclass(init=False)
class Arguments(Dataclass):
    n: int
    m: int
    seed: int | None = None
    output: Path | None = None


ARGS = parser.parse_args(namespace=Arguments())
