from collections.abc import Iterable
from typing import Any, cast, get_origin

import numpy as np
from mcda.internal.core.matrices import OutrankingMatrix
from mcda.internal.core.relations import Relation
from mcda.internal.core.values import Ranking
from mcda.relations import I, P

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


def divide_preferences(preferences: Iterable[Relation]):
    preference_relations: list[P] = []
    indifference_relations: list[I] = []
    for r in preferences:
        match r:
            case P():
                preference_relations.append(r)
            case I():
                indifference_relations.append(r)
    return preference_relations, indifference_relations


# def refused_preferences(accepted: PreferenceStructure, refused: PreferenceStructure):
#     result = refused - accepted

#     for r in result.substructure(types=[I]):
#         if P(r.a, r.b) in accepted:
#             result._relations.append(P(r.b, r.a))
#         else:
#             result._relations.append(P(r.a, r.b))

#     return result


def preference_to_numeric(r: P | I):
    a, b = (r.a, r.b) if r.a < r.b else (r.b, r.a)
    if r == P(a, b):
        return 1
    elif r == I(a, b):
        return 0
    elif r == P(b, a):
        return -1
    else:
        raise Exception()


def complementary_preference(r: P | I) -> list[P | I]:
    match r:
        case P(a=a, b=b):
            return [I(a, b), P(b, a)]
        case I(a=a, b=b):
            return [P(a, b), P(b, a)]
