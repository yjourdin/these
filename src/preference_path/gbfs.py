import heapq
from dataclasses import InitVar, field
from itertools import count, pairwise
from math import inf
from time import thread_time

from src.dataclass import dataclass

from .path_reconstructor import Node, Paths


@dataclass(order=True, slots=True)
class NodeGBFS[T](Node[T]):
    heuristic: float
    entry_count: int = field(default_factory=count().__next__, init=False)
    latest: InitVar[bool] = False

    def __post_init__(self, latest: bool):
        if latest:
            self.entry_count = -self.entry_count

    def __str__(self):
        return f"{self.item} {self.heuristic}"


@dataclass
class GBFS[T](Paths[T, NodeGBFS[T]]):
    latest: bool = False

    def init(self, sources: list[T]):
        super().init(sources)
        for i, source in enumerate(sources):
            if heuristic_value := self.heuristic(source):
                self.open_heaps[i].append(NodeGBFS(source, heuristic_value, self.latest))
            else:
                self.paths |= {i: [source]}
            heapq.heapify(self.open_heaps[i])

    def main_loop(self, max_time_loop: int):
        time_loop = 0
        while (
            (time_loop < max_time_loop)
            and (self.time < self.max_time)
            and self.open_heaps
            and any(self.open_heaps.values())
        ):
            time = thread_time()

            min_heuristic = min(heap[0].heuristic for heap in self.open_heaps.values())
            for source_id, heap in self.open_heaps.items():
                if heap[0].heuristic == min_heuristic:
                    current_node = heapq.heappop(heap)

                    # Best node
                    # current_node = heapq.heappop(self.open_heap)
                    current = current_node.item

                    if self.verbose:
                        # print(
                        #     set(self.parent[current].keys()), current_node.heuristic, flush=True
                        # )
                        with self.log_writer() as log_writer:
                            log_writer.writerow(
                                self.LogFields(
                                    Item=current,
                                    Heuristic=current_node.heuristic,
                                    Time=self.time,
                                )
                            )

                    # Explore neighborhood
                    for neighbor in self.neighborhood(current_node.item):
                        if neighbor not in self.parent:
                            self.parent[neighbor] = dict.fromkeys(self.parent[current], current)

                            # Stop when target reached
                            if (heuristic_value := self.heuristic(neighbor)) == 0:
                                paths = self.paths_from(neighbor)
                                self.paths |= paths
                                for source in self.paths:
                                    self.open_heaps.pop(source, None)
                                return self.paths
                            elif heuristic_value < inf:
                                # Add neighbor to queue
                                heapq.heappush(
                                    self.open_heaps[source_id],
                                    NodeGBFS(neighbor, heuristic_value, self.latest),
                                )

                        elif (neighbor_source_ids := set(self.parent[neighbor].keys())) != (
                            current_source_ids := set(self.parent[current].keys())
                        ):
                            # Remonte le path de current
                            if new_ids := neighbor_source_ids - current_source_ids:
                                paths = self.paths_from(current)
                                for i in new_ids:
                                    for path in paths.values():
                                        for u, v in pairwise([neighbor] + path):
                                            self.parent[v] |= {i: u}
                            # Remonte le path de neighbor
                            if new_ids := current_source_ids - neighbor_source_ids:
                                paths = self.paths_from(neighbor)
                                for i in new_ids:
                                    for path in paths.values():
                                        for u, v in pairwise([current] + path):
                                            self.parent[v] |= {i: u}
                                if neighbors_source_found_ids := (
                                    neighbor_source_ids & self.paths.keys()
                                ):
                                    paths = self.paths_from(
                                        self.paths[neighbors_source_found_ids.pop()][0]
                                    )
                                    self.paths |= paths
                                    for source in self.paths:
                                        self.open_heaps.pop(source, None)
                                    return self.paths

            # Update time
            time_loop += thread_time() - time
            self.time += thread_time() - time

        return self.paths
