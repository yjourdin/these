from pathlib import Path
from typing import cast

from mcda.internal.core.values import Ranking
from mcda.relations import PreferenceStructure
from pandas import read_csv

from src.model import GroupModel, Model
from src.models import model_from_json
from src.performance_table.normal_performance_table import NormalPerformanceTable
from src.random import rng_, seed_
from src.utils import add_filename_suffix, file_or_stdout

from .args import ARGS, TypeEnum
from .generate import noisy_comparisons, random_comparisons
from .io import to_csv

# Import data
with ARGS.model.open("r") as f:
    model = model_from_json(f.read())

A = NormalPerformanceTable(read_csv(ARGS.A, header=None))

NB_DM = model.group_size if isinstance(model, GroupModel) else 1
DMS = range(NB_DM)

match ARGS.type:
    case TypeEnum.PREFERENCE_STRUCTURE:
        # Create random seeds
        seed_shuffle, seed_error = (
            (ARGS.seed_shuffle, ARGS.seed_error)
            if (ARGS.seed_shuffle is not None) and (ARGS.seed_error is not None)
            else seed_(ARGS.seed).spawn(2)
        )

        # Create preference structure
        D: list[PreferenceStructure] = []
        rng_shuffle = rng_(seed_shuffle)
        for dm in DMS:
            model_dm = (
                cast(Model, model[dm]) if isinstance(model, GroupModel) else model
            )
            if ARGS.same:
                rng_shuffle = seed_shuffle
            D.append(random_comparisons(A, model_dm, ARGS.n, rng=seed_shuffle))

        # Add errors
        rng_error = rng_(seed_error)
        if ARGS.error:
            for dm in DMS:
                D[dm] = noisy_comparisons(D[dm], ARGS.error, rng_error)
    case TypeEnum.RANKING:
        R: list[Ranking] = []
        for dm in DMS:
            R.append(model.rank(A))


# Write results
def filename(path: Path | None, dm: int):
    if path is None:
        return None
    if NB_DM == 1:
        return path
    else:
        return add_filename_suffix(path, f"_{dm}")


for dm in DMS:
    with file_or_stdout(filename(ARGS.output, dm), "w") as f:
        match ARGS.type:
            case TypeEnum.PREFERENCE_STRUCTURE:
                to_csv(D[dm], f) # pyright: ignore[reportUnknownArgumentType, reportPossiblyUnboundVariable]
            case TypeEnum.RANKING:
                R[dm].data.to_csv(f, header=False) # pyright: ignore[reportPossiblyUnboundVariable]
