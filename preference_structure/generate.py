import numpy as np
from mcda.matrices import PerformanceTable
from mcda.relations import I, P, PreferenceStructure
from mcda.internal.core.values import Ranking
from numpy.random import Generator

from abstract_model import Model


def random_comparisons(
    nb: int,
    alternatives: PerformanceTable,
    model: Model,
    rng: Generator,
) -> PreferenceStructure:
    ranking = model.rank(alternatives)
    ranking_dict = ranking.data.to_dict()
    labels = ranking.labels
    all_pairs = np.array(np.triu_indices(len(alternatives.data), 1)).transpose()
    pairs = rng.choice(all_pairs, min(nb, len(all_pairs)), replace=False)
    result = PreferenceStructure()
    for ia, ib in pairs:
        a, b = labels[ia], labels[ib]
        if ranking_dict[a] < ranking_dict[b]:
            result._relations.append(P(a, b))
        elif ranking_dict[a] == ranking_dict[b]:
            result._relations.append(I(a, b))
        else:
            result._relations.append(P(b, a))
    return result


def from_ranking(ranking: Ranking):
    ranking_dict = ranking.data.to_dict()
    labels = ranking.labels
    all_pairs = np.array(np.triu_indices(len(ranking.data), 1)).transpose()
    result = PreferenceStructure()
    for ia, ib in all_pairs:
        a, b = labels[ia], labels[ib]
        if ranking_dict[a] < ranking_dict[b]:
            result._relations.append(P(a, b))
        elif ranking_dict[a] == ranking_dict[b]:
            result._relations.append(I(a, b))
        else:
            result._relations.append(P(b, a))
    return result


def all_comparisons(
    alternatives: PerformanceTable, model: Model
) -> PreferenceStructure:
    return from_ranking(model.rank(alternatives))


def noisy_comparisons(
    comparisons: PreferenceStructure, error_rate: float, rng: Generator
) -> PreferenceStructure:
    result = PreferenceStructure()
    relations = comparisons.relations
    selected_relations = rng.choice(
        np.array(relations), int(error_rate * len(relations)), replace=False
    )
    relations = list(set(relations) - set(selected_relations))
    selected_indifferences = [r for r in selected_relations if isinstance(r, I)]
    nb_indifferences = len(selected_indifferences)
    selected_preferences = list(set(selected_relations) - set(selected_indifferences))
    nb_preferences = len(selected_preferences)
    changed_indifferences = [
        P(*rng.permutation(r.elements)) for r in selected_indifferences
    ]
    changed_preferences = [
        (
            I(*(r.elements))
            if rng.random() < (nb_indifferences / nb_preferences)
            else P(r.b, r.a)
        )
        for r in selected_preferences
    ]
    result._relations = relations + changed_indifferences + changed_preferences
    return result
