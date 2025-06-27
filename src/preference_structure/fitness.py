from typing import Any

import numpy as np
import numpy.typing as npt
from mcda.internal.core.matrices import OutrankingMatrix
from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure

from .utils import OutrankingMatrixClass, RankingSeries, outranking_numpy


def comparisons_ranking(C: PreferenceStructure, R: dict[Any, int]):
    result: list[Relation] = []
    for r in C:
        a, b = r.elements
        match r:
            case P():
                cond = R[a] < R[b]
            case I():
                cond = R[a] == R[b]
            case _:
                cond = True
        if not cond:
            result.append(r)
    return result


def comparisons_outranking(C: PreferenceStructure, O: OutrankingMatrix):
    result: list[Relation] = []
    for r in C:
        a, b = r.elements
        match r:
            case P():
                return O.cell[a, b] * (1 - O.cell[b, a])
            case I():
                cond = O.cell[a, b] * O.cell[b, a]
            case _:
                cond = True
        if not cond:
            result.append(r)
    return result


def fitness_comparisons(Co: PreferenceStructure, Ce: PreferenceStructure):
    return sum(r in Ce for r in Co) / len(Co)


def fitness_comparisons_ranking(Co: PreferenceStructure, Re: RankingSeries):
    Re_dict = Re.to_dict()

    return 1 - (len(comparisons_ranking(Co, Re_dict)) / len(Co))


def fitness_comparisons_outranking(Co: PreferenceStructure, Oe: OutrankingMatrix):
    return 1 - (len(comparisons_outranking(Co, Oe)) / len(Co))


def fitness_ranking_comparisons(Ro: RankingSeries, Ce: PreferenceStructure):
    n = len(Ro)
    Ro_dict = Ro.to_dict()

    return 1 - (len(comparisons_ranking(Ce, Ro_dict)) / (n * (n - 1) / 2))


def fitness_outranking_comparisons(Oo: OutrankingMatrix, Ce: PreferenceStructure):
    n = len(Oo.data)

    return 1 - (len(comparisons_outranking(Ce, Oo)) / (n * (n - 1) / 2))


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
            return fitness_comparisons_outranking(o, e)
        else:
            return fitness_comparisons_ranking(o, e)
    elif isinstance(o, OutrankingMatrixClass):
        if isinstance(e, PreferenceStructure):
            return fitness_outranking_comparisons(o, e)
        else:
            return fitness_outranking(o, e)
    else:
        if isinstance(e, PreferenceStructure):
            return fitness_ranking_comparisons(o, e)
        else:
            return fitness_outranking(o, e)
