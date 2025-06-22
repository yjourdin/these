from collections.abc import Iterable
from itertools import combinations
from typing import Any

import numpy as np
from mcda.internal.core.relations import Relation
from mcda.internal.core.values import Ranking
from mcda.relations import I, P, PreferenceStructure

from ..model import Model
from ..performance_table.dominance_relation import (
    dominance_structure,
    is_subset,
)
from ..performance_table.type import PerformanceTableType
from ..random import RNGParam, rng_
from ..weak_order import WeakOrder


def random_preference_relation(
    performance_table: PerformanceTableType, rng: RNGParam = None, delta: float = 0.01
):
    dom_struct = dominance_structure(performance_table)
    pref_struct = dom_struct.copy()

    while is_subset(dom_struct, pref_struct):
        pref_struct = WeakOrder.random(performance_table.alternatives, rng).structure

    return pref_struct


def preference_relation_generator(
    ranking: Ranking,
    pairs: Iterable[tuple[Any, Any]] | None = None,
    rng: RNGParam = None,
):
    pairs = pairs if pairs is not None else combinations(ranking.labels, 2)  # type: ignore
    ranks = ranking.data.to_dict()

    if rng:
        pairs = rng_(rng).permutation(np.array(list(pairs)))  # type: ignore

    for a, b in pairs:
        if ranks[a] < ranks[b]:
            yield P(a, b)
        elif ranks[a] > ranks[b]:
            yield P(b, a)
        else:
            yield I(a, b)


def random_comparisons(
    alternatives: PerformanceTableType,
    model: Model | None,
    nb: int | None = None,
    pairs: Iterable[tuple[Any, Any]] | None = None,
    rng: RNGParam = None,
    remove_dominance: bool = False,
) -> PreferenceStructure:
    dom_struct = (
        dominance_structure(alternatives) if remove_dominance else PreferenceStructure()
    )
    if model:
        result: list[Relation] = []
        ranking = model.rank(alternatives)
        for r in preference_relation_generator(ranking, pairs, rng):
            if (not remove_dominance) or (r not in dom_struct):
                result.append(r)
            if len(result) == nb:
                break
        pref_struct = PreferenceStructure(result, validate=False)
    else:
        assert rng
        pref_struct = random_preference_relation(alternatives, rng)
        if remove_dominance:
            pref_struct -= dom_struct

    return pref_struct


def noisy_comparisons(
    comparisons: PreferenceStructure, error_rate: float, rng: RNGParam
):
    rng = rng_(rng)
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

    return PreferenceStructure(
        relations + changed_indifferences + changed_preferences, validate=False
    )
