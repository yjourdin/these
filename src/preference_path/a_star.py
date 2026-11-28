import heapq
from dataclasses import InitVar
from itertools import count, pairwise
from math import inf

from src.dataclass import dataclass, field

from ..utils import catchtime
from .path_reconstructor import Node, Paths


@dataclass(order=True, slots=True)
class NodeAstar[T](Node[T]):
    cost: float = field(compare=False)
    heuristic: float = field(compare=False)
    f: float = field(init=False)
    entry_count: int = field(default_factory=count().__next__, init=False)
    latest: InitVar[bool] = False

    def __post_init__(self, latest: bool):
        self.f = self.cost + self.heuristic
        # self.f = self.heuristic
        if latest:
            self.entry_count = -self.entry_count

    def __str__(self):
        return f"{self.item} {self.cost} {self.heuristic}"


@dataclass
class Astar[T](Paths[T, NodeAstar[T]]):
    latest: bool = False

    def init(self, sources: list[T]):
        super().init(sources)
        for i, source in enumerate(sources):
            node = NodeAstar(source, 0, self.heuristic(source), self.latest)
            self.best[i] = node
            if node.heuristic:
                self.open_heaps[i].append(node)
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
            with catchtime() as time:
                # min_f = min(heap[0].f for heap in self.open_heaps.values() if heap)
                for source_id, heap in self.open_heaps.items():
                    if heap:  # and (heap[0].f == min_f):
                        current_node = heapq.heappop(heap)

                        # Best node
                        # current_node = heapq.heappop(self.open_heap)
                        current = current_node.item

                        if self.verbose:
                            # print(
                            #     set(self.parent[current].keys()),
                            #     current_node.heuristic,
                            #     current_node.cost,
                            #     flush=True,
                            # )
                            with self.log_writer() as log_writer:
                                log_writer.writerow(
                                    self.LogFields(
                                        Item=current,
                                        Heuristic=current_node.heuristic,
                                        Cost=current_node.cost,
                                        Time=current_node.entry_count,
                                    )
                                )

                        # Explore neighborhood
                        for neighbor in self.neighborhood(current):
                            if neighbor not in self.parent:
                                self.parent[neighbor] = dict.fromkeys(
                                    self.parent[current], current
                                )

                                # Stop when target reached
                                neighbor_node = NodeAstar(
                                    neighbor,
                                    current_node.cost + 1,
                                    self.heuristic(neighbor),
                                    self.latest,
                                )
                                if (
                                    heuristic_value := neighbor_node.heuristic
                                ) < self.best[source_id].heuristic:
                                    self.best[source_id] = neighbor_node
                                if heuristic_value == 0:
                                    paths = self.paths_from(neighbor)
                                    self.paths |= paths
                                    # for path in paths.values():
                                    #     self.found[path[-1]] = neighbor
                                    for source in self.paths:
                                        self.open_heaps.pop(source, None)
                                    return self.paths
                                if heuristic_value < inf:
                                    # Add neighbor to queue
                                    heapq.heappush(
                                        self.open_heaps[source_id],
                                        neighbor_node,
                                    )
                            elif (
                                neighbor_source_ids := set(self.parent[neighbor].keys())
                            ) != (
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
                                            self.paths[
                                                neighbors_source_found_ids.pop()
                                            ][0]
                                        )
                                        self.paths |= paths
                                        for source in self.paths:
                                            self.open_heaps.pop(source, None)
                                        return self.paths

            # Update time
            time_loop += time()
            self.time += time()

        if (
            (self.time >= self.max_time)
            or (not self.open_heaps)
            or (not any(self.open_heaps.values()))
        ):
            for source_id in self.open_heaps:
                if source_id not in self.paths:
                    paths = self.paths_from(self.best[source_id].item)
                    self.paths |= paths

        return self.paths
