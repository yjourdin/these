from typing import Any, cast, get_origin

import numpy as np
from mcda.internal.core.matrices import OutrankingMatrix
from mcda.internal.core.values import Ranking
from mcda.relations import I, P, PreferenceStructure

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


def divide_preferences(preference_structure: PreferenceStructure):
    preference_relations = PreferenceStructure()
    indifference_relations = PreferenceStructure()
    for r in preference_structure:
        match r:
            case P():
                preference_relations._relations.append(r)
            case I():
                indifference_relations._relations.append(r)
    return preference_relations, indifference_relations

def refused_preferences(accepted: PreferenceStructure, refused: PreferenceStructure):
    result = refused - accepted
    
    for r in refused.substructure(types=[I]):
        if P(r.a, r.b) in accepted:
            result._relations.append(P(r.b, r.a))
        else:
            result._relations.append(P(r.a, r.b))
    
    return result