from itertools import combinations, product
from math import log
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from more_itertools import powerset

from .capacity import Capacity

Element = frozenset[Any]


class ImportanceRelation:
    def __init__(
        self,
        data: npt.NDArray[np.bool_] | int,
        labels: list[Element] = [],
        check: bool = False,
    ):
        if isinstance(data, int):
            data = np.identity(2**data, np.bool_)
        self.data = data
        self.labels = labels or [
            frozenset(s) for s in powerset(range(int(log(len(self.data), 2))))
        ]
        self.index_dict = {label: i for i, label in enumerate(self.labels)}
        if check:
            self.correct()
        self.compute_large()
        self.compute_strict()
        self.compute_eq()

    def is_reflexive(self) -> bool:
        return bool(self.data.diagonal().all())

    def correct_reflexive(self):
        np.fill_diagonal(self.data, True)

    def is_transitive(self) -> bool:
        return self.data == self.data @ self.data

    def correct_transitive(self):
        for _ in range(len(self.data)):
            self.data |= (self.data @ self.data)

    def is_complete(self) -> bool:
        return bool(np.all(self.data | self.data.transpose()))

    def correct_complete(self):
        for a, b in combinations(self.labels, 2):
            if (not self[a, b]) and (not self[b, a]):
                self[a, b] = True
                self[b, a] = True

    def is_monotonic(self) -> bool:
        for a, b in combinations(self.labels, 2):
            if a > b and not self[a, b]:
                return False
        return True

    def correct_monotonic(self):
        for a, b in combinations(self.labels, 2):
            if a > b:
                self[a, b] = True
            if b > a:
                self[b, a] = True

    def check(self):
        return (
            self.is_reflexive()
            and self.is_monotonic()
            and self.is_complete()
            and self.is_transitive()
        )

    def correct(self):
        self.correct_reflexive()
        self.correct_monotonic()
        self.correct_complete()
        self.correct_transitive()

    def __getitem__(self, key: tuple[Element, Element]) -> bool:
        return self.data[self.index_dict[key[0]], self.index_dict[key[1]]]

    def __setitem__(self, key: tuple[Element, Element], value: bool):
        self.data[self.index_dict[key[0]], self.index_dict[key[1]]] = value

    def index(self, labels: list[Element]):
        return [self.index_dict[label] for label in labels]

    def compute_large(self):
        self.large = self.data.copy()

    def compute_strict(self):
        self.strict = self.large & np.logical_not(self.large.transpose())

    def compute_eq(self):
        self.eq = self.large & self.large.transpose()

    def update(self):
        self.compute_large()
        self.compute_strict()
        self.compute_eq()

    def sub(
        self, rel: Literal["large", "strict", "eq"], a: list[Element], b: list[Element]
    ) -> npt.NDArray[np.bool_]:
        return getattr(self, rel)[np.ix_(self.index(a), self.index(b))]

    @classmethod
    def from_capacity(cls, capacity: Capacity):
        self = cls(int(log(len(capacity), 2)), check=False)
        for a, b in product(capacity, repeat=2):
            self[a, b] = capacity[a] >= capacity[b]
        self.update()
        return self

    def to_capacity(self) -> Capacity:
        capacity: dict[Element, float] = {}
        data = self.data.copy()
        labels = np.array(self.labels)
        index = list(range(len(data)))
        while len(index) > 0:
            data = data[np.ix_(index, index)]
            labels = labels[index]
            better = data.sum(1)
            minimals = np.where(better == better.min())[0]
            for i in minimals:
                capacity[labels[i]] = (  # type: ignore
                    1
                    if len(minimals) == len(index)
                    else (len(self.labels) - len(index)) / len(self.labels)
                )
            index = [i for i in range(len(data)) if i not in minimals]
        capacity[frozenset()] = 0
        return capacity
