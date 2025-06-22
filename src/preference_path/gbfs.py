import heapq
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import count
from time import process_time

from ..constants import DEFAULT_MAX_TIME
from ..dataclass import Dataclass
from .neighborhood import Neighborhood


@dataclass(order=True)
class Node[T](Dataclass):
    item: T = field(compare=False)
    heuristic: float
    entry_count: int = field(default_factory=count().__next__, init=False)


@dataclass
class PathsReconstructor[T](Dataclass):
    parent: dict[T, T | None]

    def __call__(self, v: T) -> list[T]:
        parent = self.parent[v]
        return [v] + self(parent) if parent is not None else [v]


@dataclass
class GBFS[T](Dataclass):
    neighborhood: Neighborhood[T]
    heuristic: Callable[[T], float]
    max_time: int = DEFAULT_MAX_TIME

    def __call__(self, source: T) -> list[T]:
        # Initialise
        start_time = process_time()
        self.time = process_time() - start_time
        # visited = {source}
        open_heap = [Node(source, self.heuristic(source))]
        closed_set = {source}
        parent: dict[T, T | None] = {source: None}

        # Stopping criterion
        while (self.time < self.max_time) and open_heap:
            # Best node
            current = heapq.heappop(open_heap)

            # print(
            #     current.heuristic,
            #     current.item.profiles[0][3],
            #     current.item.importance_relation,
            #     current.item.weights,
            # )
            # print()
            # print("Cur", current.heuristic, current.item.weights)
            # print("Cur", current.item.weights)

            # Explore neighborhood
            for neighbor in self.neighborhood(current.item):
                # print("Nei", neighbor.weights)
                if neighbor not in closed_set:
                    # print("passed")
                    parent[neighbor] = current.item

                    # Stop when target reached
                    if (heuristic_value := self.heuristic(neighbor)) == 0:
                        # print(len(closed_set))
                        return PathsReconstructor(parent)(neighbor)
                    else:
                        # Add neighbor to queue
                        heapq.heappush(open_heap, Node(neighbor, heuristic_value))
                        # print("Nei", heuristic_value, neighbor.weights)
                        closed_set.add(neighbor)
                        # visited.add(neighbor)

            # Update time
            self.time = process_time() - start_time

        if not open_heap:
            raise Exception("Target unreachable")

        return []
