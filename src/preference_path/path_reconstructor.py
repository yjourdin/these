from collections import defaultdict
import csv
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NotRequired, TypedDict

from src.constants import DEFAULT_MAX_TIME
from src.dataclass import Dataclass, dataclass, field
from src.utils import file_or_stdout

from .neighborhood import Neighborhood


@dataclass(order=True, slots=True)
class Node[T](Dataclass):
    item: T = field(compare=False)
    # entry_count: int = field(default_factory=count().__next__, init=False)


@dataclass
class Paths[T, N: Node[Any]](Dataclass):
    neighborhood: Neighborhood[T]
    heuristic: Callable[[T], float]
    max_time: int = DEFAULT_MAX_TIME
    verbose: bool = False
    log_path: Path | None = None

    class LogFields[S](TypedDict):
        Item: S
        Heuristic: float
        Cost: NotRequired[float]
        Time: float

    @contextmanager
    def log_writer(self):
        with file_or_stdout(self.log_path, "w", "") as f:
            yield csv.DictWriter(
                f,
                list(self.LogFields.__annotations__.keys()),
                dialect="unix",
            )

    def paths_from(self, v: T, seen: set[T] | None = None):
        seen = {v}
        result: dict[int, list[T]] = {i: [v] for i in self.parent[v]}
        finished = set()
        while continuing := (result.keys() - finished):
            for i in continuing:
                path = result[i]
                x = path[-1]
                for j, y in self.parent[x].items():
                    if not y:
                        finished.add(j)
                    elif y not in seen:
                        seen.add(y)
                        result[j] = path + [y]
        return result

        # seen = seen or set()
        # seen.add(v)
        # result: dict[int, list[T]] = {}
        # for i, parent in self.parent[v].items():
        #     if parent and parent not in seen:
        #         result |= {
        #             j: [v] + l for (j, l) in self.paths_from(parent, seen).items()
        #         }
        #     else:
        #         result |= {i: [v]}
        # return result

    def init(self, sources: list[T]):
        self.time = 0
        self.open_heaps: dict[int, list[N]] = defaultdict(list)
        self.parent: dict[T, dict[int, T | None]] = {
            source: {i: None} for i, source in enumerate(sources)
        }
        self.paths: dict[int, list[T]] = {}

        if self.verbose:
            with self.log_writer() as log_writer:
                log_writer.writeheader()

    def main_loop(self, max_time_loop: int) -> dict[int, list[T]]: ...

    def __call__(self, sources: list[T]):
        self.init(sources)

        return self.main_loop(self.max_time)
