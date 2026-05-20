import argparse
from dataclasses import dataclass
from pathlib import Path

from src.constants import DEFAULT_MAX_TIME

from ..dataclass import Dataclass

parser = argparse.ArgumentParser()
parser.add_argument("A", type=Path, help="Alternatives")
parser.add_argument("D", type=Path, help="Comparisons")
parser.add_argument("models", nargs="+", type=Path, help="Collective models")
parser.add_argument(
    "--max-time", type=int, default=DEFAULT_MAX_TIME, help="Time limit (in seconds)"
)
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument("-R", "--refused", type=Path, help="Refused comparisons")
parser.add_argument("--model-output", type=Path, help="Output model files prefix")
parser.add_argument("-o", "--output", type=Path, help="Output preference files prefix")
parser.add_argument("-r", "--result", type=Path, help="Result file")


@dataclass(init=False)
class Arguments(Dataclass):
    A: Path
    D: Path
    models: list[Path]
    max_time: int
    seed: int | None = None
    refused: Path | None = None
    model_output: Path | None = None
    output: Path | None = None
    result: Path | None = None


ARGS = parser.parse_args(namespace=Arguments())
