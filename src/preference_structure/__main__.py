from pathlib import Path
from sys import stdout

from mcda.relations import PreferenceStructure
from numpy.random import SeedSequence, default_rng
from pandas import read_csv

from ..model import GroupModel
from ..models import model_from_json
from ..performance_table.normal_performance_table import NormalPerformanceTable
from .argument_parser import parse_args
from .generate import all_comparisons, noisy_comparisons, random_comparisons
from .io import to_csv

# Parse arguments
args = parse_args()


# Import data
model = model_from_json(args.model.read())

A = NormalPerformanceTable(read_csv(args.A, header=None))

if isinstance(model, GroupModel):
    NB_DM = model.size if isinstance(model, GroupModel) else 1
DMS = range(NB_DM)


# Create random seeds
seed_shuffle, seed_error = (
    (args.seed_shuffle, args.seed_error)
    if (args.seed_shuffle is not None) and (args.seed_error is not None)
    else SeedSequence(args.seed).spawn(2)
)


# Create preference structure
D: list[PreferenceStructure] = []
if not args.same:
    rng_shuffle = default_rng(seed_shuffle)
for dm in DMS:
    model_dm = model[dm] if isinstance(model, GroupModel) else model
    if args.n > 0:
        if args.same:
            rng_shuffle = default_rng(seed_shuffle)
        D.append(random_comparisons(args.n, A, model_dm, rng_shuffle))
    else:
        D.append(all_comparisons(A, model_dm))

# Add errors
rng_error = default_rng(seed_error)
if args.error:
    for dm in DMS:
        D[dm] = noisy_comparisons(D[dm], args.error, rng_error)


# Write results
def filename(dm):
    path = Path(args.output)
    if NB_DM == 1:
        return path
    else:
        return path.parent / (path.stem + f"_{dm}" + path.suffix)


for dm in DMS:
    with stdout if args.output == "stdout" else open(filename(dm), "w") as f:
        to_csv(D[dm], f)
