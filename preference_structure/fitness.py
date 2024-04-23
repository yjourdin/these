from typing import cast

import numpy as np
from mcda.internal.core.matrices import OutrankingMatrix
from mcda.internal.core.values import Ranking
from mcda.relations import I, P, PreferenceStructure

from .utils import OutrankingMatrixClass, RankingClass, outranking_numpy


def fitness_comparisons(Co: PreferenceStructure, Ce: PreferenceStructure) -> float:
    return sum(r in Ce for r in Co) / len(Co)


def fitness_comparisons_ranking(Co: PreferenceStructure, Re: Ranking) -> float:
    Re_dict = Re.data.to_dict()
    s: int = 0
    for r in Co:
        a, b = r.elements
        match r:
            case P():
                s += Re_dict[a] < Re_dict[b]
            case I():
                s += Re_dict[a] == Re_dict[b]
    return s / len(Co)


def fitness_comparisons_outranking(
    Co: PreferenceStructure, Oe: OutrankingMatrix
) -> float:
    s: int = 0
    for r in Co:
        a, b = r.elements
        match r:
            case P():
                s += Oe.cell[a, b] * (1 - Oe.cell[b, a])
            case I():
                s += Oe.cell[a, b] * Oe.cell[b, a]
    return s / len(Co)


def fitness_ranking_comparisons(Ro: Ranking, Ce: PreferenceStructure) -> float:
    Ro_dict = Ro.data.to_dict()
    s: int = 0
    for r in Ce:
        a, b = r.elements
        match r:
            case P():
                s += Ro_dict[a] > Ro_dict[b]
            case I():
                s += Ro_dict[a] == Ro_dict[b]
    n = len(Ro)
    return s / (n * (n - 1) / 2)


def fitness_outranking_comparisons(
    Oo: OutrankingMatrix, Ce: PreferenceStructure
) -> float:
    s: int = 0
    for r in Ce:
        a, b = r.elements
        match r:
            case P():
                s += Oo.cell[a, b] * (1 - Oo.cell[b, a])
            case I():
                s += Oo.cell[a, b] * Oo.cell[b, a]
    n = len(Oo.data)
    return s / (n * (n - 1) / 2)


def fitness_outranking_numpy(Oo: np.ndarray, Oe: np.ndarray) -> float:
    ind = np.triu_indices(Oo.shape[0], 1)

    Or = Oo & Oe

    s = np.count_nonzero((Or | Or.transpose())[ind])
    ss = np.count_nonzero((Oo | Oo.transpose())[ind])

    return s / ss


def fitness_outranking(
    Oo: OutrankingMatrix | Ranking, Oe: OutrankingMatrix | Ranking
) -> float:
    return fitness_outranking_numpy(outranking_numpy(Oo), outranking_numpy(Oe))


def fitness(
    o: PreferenceStructure | OutrankingMatrix | Ranking,
    e: PreferenceStructure | OutrankingMatrix | Ranking,
) -> float:
    if isinstance(o, PreferenceStructure):
        if isinstance(e, PreferenceStructure):
            return fitness_comparisons(o, e)
        elif isinstance(e, OutrankingMatrixClass):
            e = cast(OutrankingMatrix, e)
            return fitness_comparisons_outranking(o, e)
        elif isinstance(e, RankingClass):
            e = cast(Ranking, e)
            return fitness_comparisons_ranking(o, e)
        else:
            raise TypeError("must be PreferenceStructure, OutrankingMatrix or Ranking")
    elif isinstance(o, OutrankingMatrixClass):
        o = cast(OutrankingMatrix, o)
        if isinstance(e, PreferenceStructure):
            return fitness_outranking_comparisons(o, e)
        elif isinstance(e, (OutrankingMatrixClass, RankingClass)):
            return fitness_outranking(o, e)
        else:
            raise TypeError("must be PreferenceStructure, OutrankingMatrix or Ranking")
    elif isinstance(o, RankingClass):
        o = cast(Ranking, o)
        if isinstance(e, PreferenceStructure):
            return fitness_ranking_comparisons(o, e)
        elif isinstance(e, (OutrankingMatrixClass, RankingClass)):
            return fitness_outranking(o, e)
        else:
            raise TypeError("must be PreferenceStructure, OutrankingMatrix or Ranking")
    else:
        raise TypeError("must be PreferenceStructure, OutrankingMatrix or Ranking")
