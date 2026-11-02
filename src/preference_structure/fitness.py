import numpy as np
from mcda.internal.core.matrices import OutrankingMatrix
from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure

from .utils import OutrankingMatrixClass, RankingDict, outranking_numpy


def comparisons_ranking(C: PreferenceStructure, R: RankingDict):
    result: list[Relation] = []
    for r in C:
        match r:
            case P(a=a, b=b):
                cond = R[a] < R[b]
            case I(a=a, b=b):
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
            case P(a=a, b=b):
                cond = O.cell[a, b] * (1 - O.cell[b, a])
            case I(a=a, b=b):
                cond = O.cell[a, b] * O.cell[b, a]
            case _:
                cond = True
        if not cond:
            result.append(r)
    return result


def fitness_comparisons(Co: PreferenceStructure, Ce: PreferenceStructure):
    return sum(r in Ce for r in Co) / len(Co)


def fitness_comparisons_ranking(Co: PreferenceStructure, Re: RankingDict):
    # print(comparisons_ranking(Co, Re))
    return 1 - (len(comparisons_ranking(Co, Re)) / len(Co))


def fitness_comparisons_outranking(Co: PreferenceStructure, Oe: OutrankingMatrix):
    return 1 - (len(comparisons_outranking(Co, Oe)) / len(Co))


def fitness_ranking_comparisons(Ro: RankingDict, Ce: PreferenceStructure):
    n = len(Ro)

    return 1 - (len(comparisons_ranking(Ce, Ro)) / (n * (n - 1) / 2))


def fitness_outranking_comparisons(Oo: OutrankingMatrix, Ce: PreferenceStructure):
    n = len(Oo.data)

    return 1 - (len(comparisons_outranking(Ce, Oo)) / (n * (n - 1) / 2))


def fitness_outranking_numpy(
    Oo: np.ndarray[tuple[int, int], np.dtype[np.bool]],
    Oe: np.ndarray[tuple[int, int], np.dtype[np.bool]],
):
    ind = np.triu_indices(Oo.shape[0], 1)

    Or = np.logical_not(np.logical_xor(Oo, Oe))

    s = np.count_nonzero((Or & Or.transpose())[ind])
    ss = np.count_nonzero((Oo | Oo.transpose())[ind])

    return s / ss


def fitness_outranking(
    Oo: OutrankingMatrix | RankingDict, Oe: OutrankingMatrix | RankingDict
):
    return fitness_outranking_numpy(outranking_numpy(Oo), outranking_numpy(Oe))


def fitness(
    o: PreferenceStructure | OutrankingMatrix | RankingDict,
    e: PreferenceStructure | OutrankingMatrix | RankingDict,
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
        return fitness_ranking_comparisons(o, e)
