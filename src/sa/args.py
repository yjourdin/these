import argparse
from dataclasses import dataclass
from pathlib import Path

from src.constants import DEFAULT_MAX_TIME
from src.models import ModelEnum

from ..dataclass import Dataclass

parser = argparse.ArgumentParser()


parser.add_argument("model", type=ModelEnum, choices=ModelEnum, help="Model")
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("A", type=Path, help="Alternatives")
parser.add_argument("D", nargs="+", type=Path, help="Comparisons")

init_group = parser.add_mutually_exclusive_group(required=True)
init_group.add_argument("--T0", type=float, help="Initial temperature")
init_group.add_argument(
    "--accept", type=float, help="Acceptance rate to determine initial temperature"
)

parser.add_argument("--alpha", type=float, required=True, help="Cooling coefficient")
parser.add_argument(
    "--amp", type=float, help="Weight neighborhood amplitude", default=2
)
parser.add_argument("--L", default=1, type=int, help="Length of Markov chains")

stop_group = parser.add_mutually_exclusive_group(required=True)
stop_group.add_argument("--Tf", type=float, help="Final temperature")
stop_group.add_argument(
    "--max-time", type=int, default=DEFAULT_MAX_TIME, help="Time limit (in seconds)"
)
stop_group.add_argument("--max-it", type=int, help="Max number of iterations")
stop_group.add_argument(
    "--max-it-non-improving", type=int, help="Max number of non improving iterations"
)

parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument("--seed-init", type=int, help="Initial model random seed")
parser.add_argument("--seed-sa", type=int, help="Simulated annealing random seed")
parser.add_argument("-o", "--output", type=Path, help="Output file")
parser.add_argument("-r", "--result", type=Path, help="Result file")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
parser.add_argument("-l", "--log_path", type=Path, help="Log file")
parser.add_argument(
    "--changes", nargs="+", type=int, help="Preferences previously changed"
)
parser.add_argument("--refused", type=Path, help="Refused preferences")
parser.add_argument("--accepted", type=Path, help="Accepted preferences")
parser.add_argument("--past", nargs="+", type=Path, help="Past refused preferences")
parser.add_argument("--nb-cpus", default=1, type=int, help="Number of CPUs")


@dataclass(init=False)
class Arguments(Dataclass):
    model: ModelEnum
    k: int
    A: Path
    D: list[Path]
    alpha: float
    amp: float
    L: int
    max_time: int
    nb_cpus: int
    T0: float | None = None
    accept: float | None = None
    Tf: float | None = None
    max_it: int | None = None
    max_it_non_improving: int | None = None
    seed: int | None = None
    seed_init: int | None = None
    seed_sa: int | None = None
    output: Path | None = None
    result: Path | None = None
    log_path: Path | None = None
    changes: list[int] | None = None
    refused: Path | None = None
    accepted: Path | None = None
    past: list[Path] | None = None
    verbose: bool = False


ARGS = parser.parse_args(namespace=Arguments())
