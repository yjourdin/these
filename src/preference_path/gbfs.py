import heapq
from collections.abc import Callable
from itertools import count, pairwise
from time import thread_time
from typing import cast

from src.constants import DEFAULT_MAX_TIME
from src.dataclass import Dataclass, dataclass, field
from src.utils import CustomException

from .neighborhood import Neighborhood
from .path_reconstructor import Paths


@dataclass(order=True, slots=True)
class Node[T](Dataclass):
    item: T = field(compare=False)
    heuristic: float
    entry_count: int = field(default_factory=count().__next__, init=False)


@dataclass
class GBFS[T](Paths[T]):
    neighborhood: Neighborhood[T]
    heuristic: Callable[[T], float]
    max_time: int = DEFAULT_MAX_TIME

    def init(self, sources: list[T]):
        self.start_time = thread_time()
        self.time = thread_time() - self.start_time
        self.open_heap = [Node(source, self.heuristic(source)) for source in sources]
        heapq.heapify(self.open_heap)
        self.parent = {source: {i: None} for i, source in enumerate(sources)}

    def main_loop(self) -> dict[int, list[T]]:
        while (self.time < self.max_time) and self.open_heap:
            # Best node
            current_node = heapq.heappop(self.open_heap)
            current = current_node.item
            # self.heuristic(current.item, verbose=True)

            # print(
            # current.heuristic,
            # current.item.profiles[0],
            # current.item.importance_relation[1:3],
            # current.item.weights,
            # )
            # print()
            # print("Cur", current.heuristic, current.item.weights)
            # print("Cur", current.item.weights)

            # Explore neighborhood
            for neighbor in self.neighborhood(current_node.item):
                # print("Nei", neighbor.weights)
                if neighbor not in self.parent:
                    # print("passed")
                    self.parent[neighbor] = {id: current for id in self.parent[current]}

                    # Stop when target reached
                    if (heuristic_value := self.heuristic(neighbor)) == 0:
                        # print(len(closed_set))
                        return self.paths(neighbor)
                    else:
                        # Add neighbor to queue
                        heapq.heappush(self.open_heap, Node(neighbor, heuristic_value))
                        # print("Nei", heuristic_value, neighbor.weights)

                elif (
                    neighbor_source_ids := frozenset(self.parent[neighbor].keys())
                ) != (current_source_ids := frozenset(self.parent[current].keys())):
                    # Remonte le path de neighbor
                    if new_ids := current_source_ids - neighbor_source_ids:
                        paths = self.paths(neighbor)
                        for i in new_ids:
                            for u, v in pairwise([current] + paths[i]):
                                self.parent[v] |= {i: u}
                    # Remonte le path de current
                    if new_ids := neighbor_source_ids - current_source_ids:
                        paths = self.paths(current)
                        for i in new_ids:
                            for u, v in pairwise([neighbor] + paths[i]):
                                self.parent[v] |= {i: u}

            # Update time
            self.time = thread_time() - self.start_time

        if not self.open_heap:
            raise CustomException("Target unreachable")

        return {}

    def __call__(self, sources: list[T]):
        self.init(sources)

        return self.main_loop()
