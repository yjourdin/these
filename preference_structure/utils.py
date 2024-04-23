from typing import Any, cast, get_origin

import numpy as np
from mcda.internal.core.matrices import OutrankingMatrix
from mcda.internal.core.values import Ranking

OutrankingMatrixClass = cast(Any, get_origin(OutrankingMatrix))
RankingClass = cast(Any, get_origin(Ranking))


def outranking_numpy_from_outranking(outranking: OutrankingMatrix) -> np.ndarray:
    return outranking.data.to_numpy().astype(bool, copy=False)


def outranking_numpy_from_ranking(ranking: Ranking) -> np.ndarray:
    ranking_numpy = ranking.data.to_numpy()

    return np.less_equal.outer(ranking_numpy, ranking_numpy).astype(bool, copy=False)


def outranking_numpy(o: OutrankingMatrix | Ranking) -> np.ndarray:
    if isinstance(o, OutrankingMatrixClass):
        o = cast(OutrankingMatrix, o)
        return outranking_numpy_from_outranking(o)
    elif isinstance(o, RankingClass):
        o = cast(Ranking, o)
        return outranking_numpy_from_ranking(o)
    else:
        raise TypeError("must be OutrankingMatrix or Ranking")
