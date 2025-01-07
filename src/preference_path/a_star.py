import heapq
from collections import defaultdict
from collections.abc import Callable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass, field
from itertools import count

from ..dataclass import Dataclass
from .neighborhood import Neighborhood


@dataclass(order=True)
class Node[T](Dataclass):
    heuristic: float
    entry_count: int = field(default_factory=count().__next__, init=False)
    item: T = field(compare=False)


@dataclass
class PathsReconstructor[T](Dataclass):
    parents: Mapping[T, Sequence[T]]

    def __call__(self, v: T) -> list[list[T]]:
        return (
            [[v] + path for parent in self.parents[v] for path in self(parent)]
            if self.parents[v]
            else [[v]]
        )


@dataclass
class A_star[T](Dataclass):
    neighborhood: Neighborhood[T]

    def __call__(
        self, source: T, target: float, heuristic: Callable[[T], float]
    ) -> list[list[T]]:
        open_heap = [Node(heuristic(source), source)]
        open_set: set[T] = {source}
        closed_set: set[T] = set()
        parents: defaultdict[T, MutableSequence[T]] = defaultdict(list)
        parents[source]

        while open_set:
            current = heapq.heappop(open_heap)
            open_set.remove(current.item)

            if current.heuristic == target:
                return PathsReconstructor(parents)(current.item)

            closed_set.add(current.item)

            for neighbor in self.neighborhood(current.item):
                if (neighbor not in closed_set) and (neighbor not in open_set):
                    parents[neighbor].append(current.item)
                    heapq.heappush(open_heap, Node(heuristic(neighbor), neighbor))
                    open_set.add(neighbor)

        raise Exception("Target unreachable")
