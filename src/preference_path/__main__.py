import csv
from pathlib import Path

from pandas import read_csv

from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..preference_structure.io import from_csv, to_csv
from ..srmp.model import SRMPModel
from ..utils import add_filename_suffix
from .argument_parser import parse_args
from .main import compute_model_path, compute_preference_path

# Parse arguments
args = parse_args()


# Import data
Mc = SRMPModel.from_json(args.model.read())

A = NormalPerformanceTable(read_csv(args.A, header=None))

D = from_csv(args.D)

R = [from_csv(R_file) for R_file in args.R] if args.R is not None else []


# Compute model path
model_path, time = compute_model_path(Mc, D, A, args.seed, args.max_time)


# Write model path
if args.model_output:
    for index, model in enumerate(model_path):
        filename = add_filename_suffix(Path(args.model_output), f"_{index}")

        with filename.open("w") as f:
            f.write(model.to_json())


# Compute path
path = compute_preference_path(model_path, D, A, R)


# Write path
if args.output:
    for index, preferences in enumerate(path):
        filename = add_filename_suffix(Path(args.output), f"_{index}")

        with filename.open("w") as f:
            to_csv(preferences, f)


# Write results
writer = csv.writer(args.result, "unix")
writer.writerow([len(path), len(model_path)])
