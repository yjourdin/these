import argparse
from dataclasses import dataclass
from pathlib import Path

from ..dataclass import Dataclass

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument("-o", "--output", type=Path, help="Output file")


@dataclass(init=False)
class Arguments(Dataclass):
    seed: int | None = None
    output: Path | None = None


ARGS = parser.parse_args(namespace=Arguments())
