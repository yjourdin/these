from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from ..dataclass import Dataclass
from ..performance_table.normal_performance_table import NormalPerformanceTable
from ..rmp.importance_relation import ImportanceRelation


@dataclass
class PerturbProfile(Dataclass):
    amp: float

    def __call__(self, profiles: NormalPerformanceTable, rng: Generator):
        profiles_numpy = profiles.data.to_numpy()

        return NormalPerformanceTable(
            np.sort(rng.uniform(
                np.maximum(profiles_numpy - self.amp, 0),
                np.minimum(profiles_numpy + self.amp, 1),
            ), 0)
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
    nb: int

    def __call__(self, lex_order: list[int], rng: Generator):
        lex_order = deepcopy(lex_order)

        for _ in range(self.nb):
            ind = rng.choice(len(lex_order) - 1)

            lex_order[ind], lex_order[ind + 1] = lex_order[ind + 1], lex_order[ind]

        return lex_order
