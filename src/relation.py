from functools import total_ordering
from itertools import combinations
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from mcda.relations import I, P, PreferenceStructure
from numpy.random import Generator

from .generate_weak_order import random_ranking_with_tie


@total_ordering
class Relation[Element]:
    def __init__(
        self,
        data: npt.NDArray[np.bool_] | int,
        labels: list[Element] = [],
        validate: bool = False,
    ):
        if isinstance(data, int):
            data = np.zeros((data, data), bool)
        self.data = data
        self.labels = labels or self.default_labels(self.data)
        self.index_dict = {label: i for i, label in enumerate(self.labels)}
        if validate:
            self.correct()

    @property
    def large(self):
        return self.data

    @property
    def strict(self):
        return self.large & np.logical_not(self.large.transpose())

    @property
    def eq(self):
        return self.large & self.large.transpose()

    def __getitem__(self, key: tuple[Element, Element]) -> bool:
        return self.data[self.index_dict[key[0]], self.index_dict[key[1]]]

    def __setitem__(self, key: tuple[Element, Element], value: bool):
        self.data[self.index_dict[key[0]], self.index_dict[key[1]]] = value

    def sub(
        self, rel: Literal["large", "strict", "eq"], a: list[Element], b: list[Element]
    ) -> npt.NDArray[np.bool_]:
        return getattr(self, rel)[np.ix_(self.index(a), self.index(b))]

    @staticmethod
    def default_labels(data: npt.NDArray[np.bool_]) -> list[Element]: ...

    def check(self) -> bool:
        return True

    def correct(self): ...

    def index(self, labels: list[Element]):
        return [self.index_dict[label] for label in labels]

    def __eq__(self, other):
        return np.all(self.large == other.large)

    def __lt__(self, other):
        return (self.large & other.large) == self.large

    @property
    def structure(self):
        preference_structure = PreferenceStructure()
        for a, b in np.transpose(np.nonzero(self.strict)).tolist():
            preference_structure._relations.append(P(a, b))
        for a, b in np.transpose(np.nonzero(self.eq)).tolist():
            preference_structure._relations.append(I(a, b))
        return preference_structure


class ReflexiveRelation(Relation):
    @staticmethod
    def check_reflexive(relation: Relation):
        return bool(relation.data.diagonal().all())

    def check(self):
        return super().check() and self.check_reflexive(self)

    def correct(self):
        super().correct()
        np.fill_diagonal(self.data, True)


class TransitiveRelation(Relation):
    @staticmethod
    def check_transitive(relation: Relation):
        return relation.data == relation.data @ relation.data

    def check(self):
        return super().check() and self.check_transitive(self)

    def correct(self):
        super().correct()
        self.data |= np.linalg.multi_dot([self.data] * len(self.data))


class CompleteRelation(Relation):
    @staticmethod
    def check_complete(relation: Relation):
        return bool(np.all(relation.data | relation.data.transpose()))

    def check(self):
        return super().check() and self.check_complete(self)

    def correct(self):
        super().correct()
        for a, b in combinations(self.labels, 2):
            if (not self[a, b]) and (not self[b, a]):
                self[a, b] = True
                self[b, a] = True


class Preorder(TransitiveRelation, ReflexiveRelation): ...


class WeakOrder[Element](CompleteRelation, Preorder):
    @classmethod
    def random(cls, labels: list[Element], rng: Generator, **kwargs):
        ranking = random_ranking_with_tie(labels, rng, **kwargs)

        return cls(np.less_equal.outer(ranking, ranking), labels)

    @classmethod
    def random_from_ranking(cls, ranking: npt.NDArray[np.int_], labels: list[Element]):
        return cls(np.less_equal.outer(ranking, ranking), labels)


class MonotonicRelation(Relation[frozenset[Any]]):
    @staticmethod
    def check_monotonic(relation: Relation[frozenset[Any]]):
        result = True
        for a, b in combinations(relation.labels, 2):
            if (a > b and not relation[a, b]) or (b > a and not relation[b, a]):
                return False
        return result

    def check(self):
        return super().check() and self.check_monotonic(self)

    def correct(self):
        super().correct()
        for a, b in combinations(self.labels, 2):
            if a > b:
                self[a, b] = True
            if b > a:
                self[b, a] = True
