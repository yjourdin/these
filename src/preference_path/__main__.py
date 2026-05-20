import csv

from more_itertools import join_mappings
from pandas import read_csv

from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.preference_structure.io import from_csv, to_csv
from src.srmp.model import SRMPModel
from src.utils import add_filename_suffix, file_or_stdout

from .args import ARGS
from .main import compute_model_paths, compute_preference_path

# Import data
Mc: list[SRMPModel] = []
for model in ARGS.models:
    with model.open("r") as f:
        Mc.append(SRMPModel.from_json(f.read()))

A = NormalPerformanceTable(read_csv(ARGS.A, header=None))

with ARGS.D.open("r") as f:
    D = from_csv(f)

Refused = None
if ARGS.refused:
    with ARGS.refused.open("r") as f:
        Refused = from_csv(f)


# Compute model path
model_paths, time = compute_model_paths(Mc, D, A, ARGS.seed, ARGS.max_time)


# Write model path
if ARGS.model_output:
    for i, path in model_paths.items():
        for j, model in enumerate(path):
            filename = add_filename_suffix(ARGS.model_output, f"_{i}_{j}")

            with filename.open("w") as f:
                f.write(model.to_json())


# Compute path
paths = {
    i: compute_preference_path(model_path, D, A, Refused)
    for i, model_path in model_paths.items()
}


# Write path
if ARGS.output:
    for i, path in paths.items():
        for j, preferences in enumerate(path):
            filename = add_filename_suffix(ARGS.output, f"_{i}_{j}")

            with filename.open("w") as f:
                to_csv(preferences, f)


# Write results
if ARGS.result:
    with file_or_stdout(ARGS.result, "w", "") as f:
        writer = csv.writer(f, "unix")
        writer.writerows([
            [i] + list(x.values())
            for i, x in join_mappings(path=paths, model_path=model_paths).items()
        ])
