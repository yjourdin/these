from typing import cast

import numpy as np
import numpy.typing as npt
from mcda.internal.core.matrices import OutrankingMatrix
from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure
from pandas import Series

from .utils import OutrankingMatrixClass, RankingSeries, outranking_numpy


def fitness_comparisons(Co: PreferenceStructure, Ce: PreferenceStructure):
    return sum(r in Ce for r in Co) / len(Co)


def fitness_comparisons_ranking(Co: PreferenceStructure, Re: RankingSeries):
    Re_dict = Re.to_dict()

    def correct(r: Relation):
        a, b = r.elements
        match r:
            case P():
                return Re_dict[a] < Re_dict[b]
            case I():
                return Re_dict[a] == Re_dict[b]
            case _:
                return False

    return sum(map(correct, Co)) / len(Co)


def fitness_comparisons_outranking(Co: PreferenceStructure, Oe: OutrankingMatrix):
    def correct(r: Relation):
        a, b = r.elements
        match r:
            case P():
                return Oe.cell[a, b] * (1 - Oe.cell[b, a])
            case I():
                return Oe.cell[a, b] * Oe.cell[b, a]
            case _:
                return False

    return sum(map(correct, Co)) / len(Co)


def fitness_ranking_comparisons(Ro: RankingSeries, Ce: PreferenceStructure):
    n = len(Ro)
    Ro_dict = Ro.to_dict()

    def correct(r: Relation):
        a, b = r.elements
        match r:
            case P():
                return Ro_dict[a] < Ro_dict[b]
            case I():
                return Ro_dict[a] == Ro_dict[b]
            case _:
                return False

    return sum(map(correct, Ce)) / (n * (n - 1) / 2)


def fitness_outranking_comparisons(Oo: OutrankingMatrix, Ce: PreferenceStructure):
    n = len(Oo.data)

    def correct(r: Relation):
        a, b = r.elements
        match r:
            case P():
                return Oo.cell[a, b] * (1 - Oo.cell[b, a])
            case I():
                return Oo.cell[a, b] * Oo.cell[b, a]
            case _:
                return False

    return sum(map(correct, Ce)) / (n * (n - 1) / 2)


def fitness_outranking_numpy(Oo: npt.NDArray[np.bool_], Oe: npt.NDArray[np.bool_]):
    ind = np.triu_indices(Oo.shape[0], 1)

    Or = np.logical_not(np.logical_xor(Oo, Oe))

    s = np.count_nonzero((Or & Or.transpose())[ind])
    ss = np.count_nonzero((Oo | Oo.transpose())[ind])

    return s / ss


def fitness_outranking(
    Oo: OutrankingMatrix | RankingSeries, Oe: OutrankingMatrix | RankingSeries
):
    return fitness_outranking_numpy(outranking_numpy(Oo), outranking_numpy(Oe))


def fitness(
    o: PreferenceStructure | OutrankingMatrix | RankingSeries,
    e: PreferenceStructure | OutrankingMatrix | RankingSeries,
):
    if isinstance(o, PreferenceStructure):
        if isinstance(e, PreferenceStructure):
            return fitness_comparisons(o, e)
        elif isinstance(e, OutrankingMatrixClass):
            e = cast(OutrankingMatrix, e)
            return fitness_comparisons_outranking(o, e)
        elif isinstance(e, Series):
            return fitness_comparisons_ranking(o, e)
        else:
            raise TypeError("must be PreferenceStructure, OutrankingMatrix or Ranking")
    elif isinstance(o, OutrankingMatrixClass):
        o = cast(OutrankingMatrix, o)
        if isinstance(e, PreferenceStructure):
            return fitness_outranking_comparisons(o, e)
        elif isinstance(e, (OutrankingMatrixClass, Series)):
            return fitness_outranking(o, e)
        else:
            raise TypeError("must be PreferenceStructure, OutrankingMatrix or Ranking")
    elif isinstance(o, Series):
        if isinstance(e, PreferenceStructure):
            return fitness_ranking_comparisons(o, e)
        elif isinstance(e, (OutrankingMatrixClass, Series)):
            return fitness_outranking(o, e)
        else:
            raise TypeError("must be PreferenceStructure, OutrankingMatrix or Ranking")
    else:
        raise TypeError("must be PreferenceStructure, OutrankingMatrix or Ranking")
