import csv
from pathlib import Path

from more_itertools import join_mappings
from pandas import read_csv

from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.preference_structure.io import from_csv, to_csv
from src.srmp.model import SRMPModel
from src.utils import add_filename_suffix

from .args import ARGS
from .main import compute_model_paths, compute_preference_path

# Import data
Mc = [SRMPModel.from_json(model.read()) for model in ARGS.models]

A = NormalPerformanceTable(read_csv(ARGS.A, header=None))

D = from_csv(ARGS.D)

R = from_csv(ARGS.R)


# Compute model path
model_paths, time = compute_model_paths(Mc, D, A, ARGS.seed, ARGS.max_time)


# Write model path
if ARGS.model_output:
    for i, path in model_paths.items():
        for j, model in enumerate(path):
            filename = add_filename_suffix(Path(ARGS.model_output), f"_{i}_{j}")

            with filename.open("w") as f:
                f.write(model.to_json())


# Compute path
paths = {
    i: compute_preference_path(model_path, D, A, R)
    for i, model_path in model_paths.items()
}


# Write path
if ARGS.output:
    for i, path in paths.items():
        for j, preferences in enumerate(path):
            filename = add_filename_suffix(Path(ARGS.output), f"_{i}_{j}")

            with filename.open("w") as f:
                to_csv(preferences, f)


# Write results
writer = csv.writer(ARGS.result, "unix")
writer.writerows([
    [i] + list(x.values())
    for i, x in join_mappings(path=paths, model_path=model_paths).items()
])
