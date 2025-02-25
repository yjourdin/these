from collections.abc import MutableMapping, Sequence
from itertools import combinations
from typing import Any

from mcda.internal.core.relations import Relation
from mcda.relations import I, P, PreferenceStructure
from numpy.random import Generator

from .julia.function import generate_weak_order
from .random import Random, seed


class WeakOrder[Element](MutableMapping[Element, int], Random):
    def __init__(
        self, scores: Sequence[int] | None = None, labels: list[Element] | None = None
    ) -> None:
        if scores and labels:
            self.dict = dict(zip(labels, scores))

    def __getitem__(self, key: Element):
        return self.dict[key]

    def __setitem__(self, key: Element, value: int):
        self.dict[key] = value

    def __delitem__(self, key: Element):
        del self.dict[key]

    def __iter__(self):
        return iter(self.dict)

    def __len__(self):
        return len(self.dict)

    @property
    def structure(self):
        result: list[Relation] = []
        for a, b in combinations(self.dict, 2):
            diff = self.dict[a] - self.dict[b]
            if diff > 0:
                result.append(P(a, b))
            elif diff < 0:
                result.append(P(b, a))
            else:
                result.append(I(a, b))
        return PreferenceStructure(result, validate=False)

    @classmethod
    def random(cls, labels: list[Element], rng: Generator, *args: Any, **kwargs: Any):
        scores = generate_weak_order(len(labels), seed(rng))
        return cls(scores, labels)
