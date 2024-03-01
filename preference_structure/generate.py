import numpy as np
from mcda.core.matrices import PerformanceTable
from mcda.core.relations import (
    IndifferenceRelation,
    PreferenceRelation,
    PreferenceStructure,
    Relation,
)
from mcda.core.values import Ranking
from numpy.random import Generator

from model import Model


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
    pairs = rng.choice(all_pairs, nb, replace=False)
    result = PreferenceStructure()
    relations: list[Relation] = []
    for ia, ib in pairs:
        a, b = labels[ia], labels[ib]
        if ranking_dict[a] < ranking_dict[b]:
            relations.append(PreferenceRelation(a, b))
        elif ranking_dict[a] == ranking_dict[b]:
            relations.append(IndifferenceRelation(a, b))
        else:
            relations.append(PreferenceRelation(b, a))
    result._relations = relations
    return result


def from_ranking(ranking: Ranking):
    ranking_dict = ranking.data.to_dict()
    labels = ranking.labels
    all_pairs = np.array(np.triu_indices(len(ranking.data), 1)).transpose()
    result = PreferenceStructure()
    relations: list[Relation] = []
    for ia, ib in all_pairs:
        a, b = labels[ia], labels[ib]
        if ranking_dict[a] < ranking_dict[b]:
            relations.append(PreferenceRelation(a, b))
        elif ranking_dict[a] == ranking_dict[b]:
            relations.append(IndifferenceRelation(a, b))
        else:
            relations.append(PreferenceRelation(b, a))
    result._relations = relations
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
    selected_indifferences = [
        r for r in selected_relations if isinstance(r, IndifferenceRelation)
    ]
    nb_indifferences = len(selected_indifferences)
    selected_preferences = list(set(selected_relations) - set(selected_indifferences))
    nb_preferences = len(selected_preferences)
    changed_indifferences = [
        PreferenceRelation(*rng.permutation(r.elements)) for r in selected_indifferences
    ]
    changed_preferences = [
        (
            IndifferenceRelation(*(r.elements))
            if rng.random() < (nb_indifferences / nb_preferences)
            else PreferenceRelation(r.b, r.a)
        )
        for r in selected_preferences
    ]
    result._relations = relations + changed_indifferences + changed_preferences
    return result
