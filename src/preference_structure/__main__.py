from pathlib import Path
from sys import stdout
from typing import cast

from mcda.internal.core.values import Ranking
from mcda.relations import PreferenceStructure
from numpy.random import SeedSequence
from pandas import read_csv

from ..model import GroupModel, Model
from ..models import model_from_json
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..random import rng
from ..utils import add_filename_suffix
from .argument_parser import TypeEnum, parse_args
from .generate import noisy_comparisons, random_comparisons
from .io import to_csv

# Parse arguments
args = parse_args()


# Import data
model = model_from_json(args.model.read())

A = NormalPerformanceTable(read_csv(args.A, header=None))

NB_DM = model.group_size if isinstance(model, GroupModel) else 1
DMS = range(NB_DM)

match cast(TypeEnum, args.type):
    case TypeEnum.PREFERENCE_STRUCTURE:
        # Create random seeds
        seed_shuffle, seed_error = (
            (args.seed_shuffle, args.seed_error)
            if (args.seed_shuffle is not None) and (args.seed_error is not None)
            else SeedSequence(args.seed).spawn(2)
        )

        # Create preference structure
        D: list[PreferenceStructure] = []
        rng_shuffle = rng(seed_shuffle)
        for dm in DMS:
            model_dm = (
                cast(Model, model[dm]) if isinstance(model, GroupModel) else model
            )
            if args.same:
                rng_shuffle = rng(seed_shuffle)
            D.append(random_comparisons(A, model_dm, args.n, rng=rng_shuffle))

        # Add errors
        rng_error = rng(seed_error)
        if args.error:
            for dm in DMS:
                D[dm] = noisy_comparisons(D[dm], args.error, rng_error)
    case TypeEnum.RANKING:
        R: list[Ranking] = []
        for dm in DMS:
            R.append(model.rank(A))


# Write results
def filename(dm: int):
    path = Path(args.output)
    if NB_DM == 1:
        return path
    else:
        return add_filename_suffix(path, f"_{dm}")


for dm in DMS:
    with stdout if args.output == "stdout" else open(filename(dm), "w") as f:
        match cast(TypeEnum, args.type):
            case TypeEnum.PREFERENCE_STRUCTURE:
                to_csv(D[dm], f)  # type: ignore
            case TypeEnum.RANKING:
                R[dm].data.to_csv(f, header=False, index=False)  # type: ignore
