import argparse
from dataclasses import dataclass, field
from pathlib import Path

from src.constants import DEFAULT_MAX_TIME, EPSILON
from src.models import ModelEnum
from src.srmp.model import SRMPParamFlag

from ..dataclass import Dataclass

parser = argparse.ArgumentParser()
parser.add_argument("model", type=ModelEnum, choices=ModelEnum, help="Model")
parser.add_argument("k", type=int, help="Number of profiles")
parser.add_argument("A", type=Path, help="Alternatives")
parser.add_argument("D", nargs="+", type=Path, help="Comparisons")
parser.add_argument(
    "-l", "--lex-order", nargs="+", type=int, help="Lexicographic order"
)
parser.add_argument(
    "--shared",
    nargs="+",
    default=[],
    type=SRMPParamFlag,
    choices=SRMPParamFlag,
    help="Parameters shared between decision makers",
)
parser.add_argument(
    "-c", "--collective", action="store_true", help="Elicit collective model"
)
parser.add_argument("--group", action="store_true", help="Elicit group models")
parser.add_argument(
    "--changes", nargs="+", type=int, help="Preferences previously changed"
)
parser.add_argument("--refused", type=Path, help="Refused preferences")
parser.add_argument("--accepted", type=Path, help="Accepted preferences")
parser.add_argument("--reference", type=Path, help="Reference model")
parser.add_argument("--profile-amp", type=float, help="Profiles amplitude")
parser.add_argument("--weight-amp", type=float, help="Weights amplitude")
parser.add_argument("--references", nargs="+", type=Path, help="Reference models")
parser.add_argument(
    "--max-time", type=int, default=DEFAULT_MAX_TIME, help="Time limit (in seconds)"
)
parser.add_argument(
    "-g",
    "--gamma",
    default=EPSILON,
    type=float,
    help="Value used for modeling strict inequalities",
)
parser.add_argument(
    "-n",
    "--no-inconsistencies",
    action="store_true",
    help="Inconsistent comparisons will not be taken into account",
)
parser.add_argument("-o", "--output", type=Path, help="Output file")
parser.add_argument("-r", "--result", type=Path, help="Result file")
parser.add_argument("-s", "--seed", type=int, help="Random seed")
parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
parser.add_argument("--log-path", type=Path, help="Log file")
parser.add_argument("--nb-cpus", default=1, type=int, help="Number of CPUs")


@dataclass(init=False)
class Arguments(Dataclass):
    model: ModelEnum
    k: int
    A: Path
    D: list[Path]
    shared: list[SRMPParamFlag] = field(default_factory=list)
    max_time: int
    gamma: float
    nb_cpus: int
    no_inconsistencies: bool = False
    lex_order: list[int] | None = None
    collective: bool = False
    group: bool = False
    changes: list[int] | None = None
    refused: Path | None = None
    accepted: Path | None = None
    reference: Path | None = None
    profile_amp: float | None = None
    weight_amp: float | None = None
    references: list[Path] | None = None
    output: Path | None = None
    result: Path | None = None
    seed: int | None = None
    verbose: bool = False
    log_path: Path | None = None


ARGS = parser.parse_args(namespace=Arguments())
