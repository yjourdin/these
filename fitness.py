import numpy as np
from mcda.internal.core.values import Ranking
from mcda.relations import I, P, PreferenceStructure


def fitness_comparisons(ranking: Ranking, comparisons: PreferenceStructure):
    # ranking_dict = ranking.data.to_dict()
    s: int = 0
    for r in comparisons:
        a, b = r.elements
        match r:
            case P():
                # s += cast(int, ranking_dict[a]) < cast(int, ranking_dict[b])
                s += ranking[a] < ranking[b]
            case I():
                # s += cast(int, ranking_dict[a]) == cast(int, ranking_dict[b])
                s += ranking[a] == ranking[b]
    return s / len(comparisons)


def fitness_ranking(original: Ranking, elicited: Ranking):
    Ro = original.data.to_numpy()
    Re = elicited.data.to_numpy()

    outranking_o = np.less.outer(Ro, Ro).astype("int64", copy=False)
    outranking_e = np.less.outer(Re, Re).astype("int64", copy=False)

    outranking_o = outranking_o - outranking_o.transpose()
    outranking_e = outranking_e - outranking_e.transpose()

    ind = np.triu_indices(len(Ro), 1)

    return np.equal(outranking_o[ind], outranking_e[ind]).sum() / len(ind[0])
