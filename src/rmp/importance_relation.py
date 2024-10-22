from itertools import product
from math import log

import numpy as np
from more_itertools import powerset
from numpy.random import Generator

from ..relation import MonotonicRelation, WeakOrder
from .capacity import Capacity


class ImportanceRelation(MonotonicRelation, WeakOrder):
    @staticmethod
    def default_labels_from_int(i):
        return [frozenset(s) for s in powerset(range(i))]

    @staticmethod
    def default_labels(data):
        return ImportanceRelation.default_labels_from_int(int(log(len(data), 2)))

    @classmethod
    def random(cls, nb: int, rng: Generator, **kwargs):
        k = 0
        while not MonotonicRelation.check_monotonic(
            weak_order := WeakOrder.random(
                cls.default_labels_from_int(nb), rng, **kwargs
            )
        ):
            k+=1
            print(k)
        print(k)
        return cls(weak_order.data, weak_order.labels, validate=False)

    @classmethod
    def from_capacity(cls, capacity: Capacity):
        self = cls(len(capacity), validate=False)
        for a, b in product(capacity, repeat=2):
            self[a, b] = capacity[a] >= capacity[b]
        return self

    def to_capacity(self) -> Capacity:
        capacity: Capacity = {}
        data = self.large.copy()
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
