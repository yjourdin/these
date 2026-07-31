import heapq
from itertools import pairwise
from math import inf
from time import thread_time

from src.dataclass import dataclass, field

from .path_reconstructor import Node, Paths


@dataclass(order=True, slots=True)
class NodeAstar[T](Node[T]):
    cost: float = field(compare=False)
    heuristic: float = field(compare=False)
    f: float = field(init=False)

    def __post_init__(self):
        self.f = self.cost + self.heuristic

    def __str__(self) -> str:
        return f"{self.item} {self.cost} {self.heuristic}"


@dataclass
class Astar[T](Paths[T, NodeAstar[T]]):
    def init(self, sources: list[T]):
        super().init(sources)
        for i, source in enumerate(sources):
            if heuristic_value := self.heuristic(source):
                self.open_heap.append(NodeAstar(source, 0, heuristic_value))
            else:
                self.paths |= {i: [source]}
        heapq.heapify(self.open_heap)

    def main_loop(self, max_time: int):
        while (self.time < min(max_time, self.max_time)) and self.open_heap:
            time = thread_time()

            # Best node
            current_node = heapq.heappop(self.open_heap)
            current = current_node.item

            if self.verbose:
                with self.log_writer() as log_writer:
                    log_writer.writerow(
                        self.LogFields(
                            Item=current,
                            Heuristic=current_node.heuristic,
                            Cost=current_node.cost,
                            Time=self.time,
                        )
                    )

            # Explore neighborhood
            for neighbor in self.neighborhood(current):
                if neighbor not in self.parent:
                    self.parent[neighbor] = {id: current for id in self.parent[current]}

                    # Stop when target reached
                    if (heuristic_value := self.heuristic(neighbor)) == 0:
                        paths = self.paths_from(neighbor)
                        self.paths |= paths
                        # for path in paths.values():
                        #     self.found[path[-1]] = neighbor
                        return self.paths
                    elif heuristic_value < inf:
                        # Add neighbor to queue
                        heapq.heappush(
                            self.open_heap,
                            NodeAstar(neighbor, current_node.cost + 1, heuristic_value),
                        )
                elif (neighbor_source_ids := set(self.parent[neighbor].keys())) != (
                    current_source_ids := set(self.parent[current].keys())
                ):
                    # Remonte le path de current
                    if new_ids := neighbor_source_ids - current_source_ids:
                        paths = self.paths_from(current)
                        for path in paths.values():
                            for u, v in pairwise([neighbor] + path):
                                for i in new_ids:
                                    self.parent[v] |= {i: u}
                    # Remonte le path de neighbor
                    if new_ids := current_source_ids - neighbor_source_ids:
                        paths = self.paths_from(neighbor)
                        for path in paths.values():
                            for i in new_ids:
                                for u, v in pairwise([current] + path):
                                    self.parent[v] |= {i: u}
                        if neighbors_source_found_ids := (
                            neighbor_source_ids & self.paths.keys()
                        ):
                            paths = self.paths_from(
                                self.paths[neighbors_source_found_ids.pop()][0]
                            )
                            self.paths |= paths
                            return self.paths

            # Update time
            self.time += thread_time() - time

        return self.paths

    def __call__(self, sources: list[T]):
        self.init(sources)

        return self.main_loop(self.max_time)
