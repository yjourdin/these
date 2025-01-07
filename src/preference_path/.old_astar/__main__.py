from pathlib import Path

from pandas import read_csv

from src.utils import add_filename_suffix

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.io import from_csv, to_csv
from ..srmp.model import SRMPModel
from .argument_parser import parse_args
from .main import compute_preference_path

# Parse arguments
args = parse_args()


# Import data
Mc = SRMPModel.from_json(args.model.read())

A = NormalPerformanceTable(read_csv(args.A, header=None))

D = from_csv(args.D)


# Compute path
path = compute_preference_path(Mc, D, A)


# Write path
for index, preferences in enumerate(path):
    filename = add_filename_suffix(Path(args.output), f"_{index}")

    with filename.open("w") as f:
        to_csv(preferences, f)
