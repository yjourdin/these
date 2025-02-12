from copy import deepcopy
from dataclasses import InitVar, dataclass

import numpy as np
from numpy.random import Generator

from ..dataclass import Dataclass
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..utils import tolist
from .importance_relation import ImportanceRelation
from .permutation import all_max_adjacent_distance


@dataclass
class PerturbProfile(Dataclass):
    amp: float

    def __call__(self, profiles: NormalPerformanceTable, rng: Generator):
        profiles_numpy = profiles.data.to_numpy()
        return NormalPerformanceTable(
            np.sort(
                rng.uniform(
                    np.maximum(profiles_numpy - self.amp, 0),
                    np.minimum(profiles_numpy + self.amp, 1),
                ),
                0,
            )
        )


@dataclass
class PerturbImportanceRelation(Dataclass):
    nb: int

    def __call__(self, importance_relation: ImportanceRelation, rng: Generator):
        importance_relation = deepcopy(importance_relation)

        for _ in range(self.nb):
            keys = list(importance_relation)
            min_score = max_score = 0

            while min_score >= max_score:
                coalition = keys[rng.choice(len(importance_relation))]
                min_score = importance_relation.min(coalition)
                max_score = importance_relation.max(coalition)

            score = importance_relation[coalition]
            available_score = []
            if score > min_score:
                available_score.append(score - 1)
            if score < max_score:
                available_score.append(score + 1)

            score = rng.choice(available_score)
            importance_relation[coalition] = score

        return importance_relation


@dataclass
class PerturbLexOrder(Dataclass):
    k: InitVar[int]
    nb: InitVar[int]

    def __post_init__(self, k: int, nb: int):
        self.all_permutations = all_max_adjacent_distance(list(range(k)), nb)

    def __call__(self, lex_order: list[int], rng: Generator) -> list[int]:
        lex_order_numpy = np.array(lex_order, dtype=np.int_)

        permutation = list(
            tuple(self.all_permutations)[rng.choice(len(self.all_permutations))]
        )

        return tolist(lex_order_numpy[permutation])
