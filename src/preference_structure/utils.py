from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import numpy.typing as npt
from mcda.internal.core.matrices import AdjacencyValueMatrix, OutrankingMatrix
from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure
from pandas import Series

OutrankingMatrixClass = AdjacencyValueMatrix
type RankingSeries = Series[int]


def preference_structure_from_outranking(outranking: OutrankingMatrix):
    relations: list[Relation] = list()
    for ii, i in enumerate(outranking.vertices):  # type: ignore
        for j in outranking.vertices[ii + 1 :]:
            if outranking.data.at[i, j]:
                if outranking.data.at[j, i]:
                    relations.append(I(i, j))
                else:
                    relations.append(P(i, j))
            elif outranking.data.at[j, i]:
                relations.append(P(j, i))
    return PreferenceStructure(relations, validate=False)


def outranking_numpy_from_outranking(
    outranking: OutrankingMatrix,
) -> npt.NDArray[np.bool_]:
    return outranking.data.to_numpy().astype(bool, copy=False)


def outranking_numpy_from_ranking(ranking: RankingSeries):
    ranking_numpy: npt.NDArray[np.int_] = ranking.to_numpy()

    return np.less_equal.outer(ranking_numpy, ranking_numpy).astype(bool, copy=False)


def outranking_numpy(o: OutrankingMatrix | RankingSeries):
    if isinstance(o, OutrankingMatrixClass):
        return outranking_numpy_from_outranking(o)
    else:
        return outranking_numpy_from_ranking(o)


def divide_preferences(preferences: Iterable[Relation]):
    preference_relations: list[P] = []
    indifference_relations: list[I] = []
    for r in preferences:
        match r:
            case P():
                preference_relations.append(r)
            case I():
                indifference_relations.append(r)
            case _:
                raise ValueError(f"{r} is not a preference relation")
    return preference_relations, indifference_relations


# def refused_preferences(accepted: PreferenceStructure, refused: PreferenceStructure):
#     result = refused - accepted

#     for r in result.substructure(types=[I]):
#         if P(r.a, r.b) in accepted:
#             result._relations.append(P(r.b, r.a))
#         else:
#             result._relations.append(P(r.a, r.b))

#     return result


def preference_to_numeric(r: Relation):
    a, b = (r.a, r.b) if r.a < r.b else (r.b, r.a)
    if r == P(a, b):
        return 1
    elif r == I(a, b):
        return 0
    elif r == P(b, a):
        return -1
    else:
        raise Exception(f"Relation {r} not recognized")


def complementary_relation(r: P | I) -> list[P | I]:
    match r:
        case P(a=a, b=b):
            return [I(a, b), P(b, a)]
        case I(a=a, b=b):
            return [P(a, b), P(b, a)]


def complementary_preference(preferences: Iterable[Relation]) -> list[P | I]:
    relations: defaultdict[frozenset[Any], list[P | I]] = defaultdict(list)
    for r in preferences:
        if isinstance(r, P | I):
            relations[frozenset((r.a, r.b))].append(r)

    result: list[P | I] = []
    for lst in relations.values():
        result.extend(
            list(set.intersection(*[set(complementary_relation(r)) for r in lst]))  # type: ignore
        )

    return result
