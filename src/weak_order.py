from collections.abc import MutableMapping, Sequence
from itertools import combinations

from mcda.relations import I, P, PreferenceStructure
from numpy.random import Generator

from .julia.function import generate_weak_order
from .random import Random, seed


class WeakOrder[Element](MutableMapping[Element, int], Random):
    def __init__(self, scores: Sequence[int] = [], labels: list[Element] = []) -> None:
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
        preference_structure = PreferenceStructure()
        for a, b in combinations(self.dict, 2):
            diff = self.dict[a] - self.dict[b]
            if diff > 0:
                preference_structure._relations.append(P(a, b))
            elif diff < 0:
                preference_structure._relations.append(P(b, a))
            else:
                preference_structure._relations.append(I(a, b))
        return preference_structure

    @classmethod
    def random(cls, labels: list[Element], rng: Generator):
        scores = generate_weak_order(len(labels), seed(rng))
        return cls(scores, labels)
