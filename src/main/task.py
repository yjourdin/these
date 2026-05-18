from abc import abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, fields
from typing import Any, ClassVar, NamedTuple

from src.dataclass import FrozenDataclass
from src.random import SeedLike, SeedMixin, int_, seed_

from ..utils import catchtime
from .abstract_task import AbstractTask
from .csv_files import TaskFields
from .directory import Directory


class TaskResult(NamedTuple):
    res: Any
    time: float


class TaskException(Exception):
    pass


type FutureTask = Future[TaskResult]


def result_list(
    futures: list[FutureTask],
) -> list[TaskResult]:
    return [future.result() for future in futures]


def result_dict[T](
    futures: dict[T, FutureTask],
) -> dict[T, TaskResult]:
    return {k: future.result() for k, future in futures.items()}


@dataclass(frozen=True)
class Task(FrozenDataclass, AbstractTask):
    name: ClassVar[str]

    def __str__(self) -> str:
        return f"{self.name:13} ({', '.join(f'{field.name}: {getattr(self, field.name)!s}' for field in fields(self))})"

    def __call__(self, dir: Directory, *args: Any, **kwargs: Any):
        with catchtime() as time:
            result = self.task(*args, dir=dir, **kwargs)

        csv_file = dir.csv_files["tasks"]
        csv_file.writerow(**self.log(time(), *args, **kwargs))

        return TaskResult(result, time())

    @abstractmethod
    def task(self, dir: Directory, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def done(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def log(self, time: float, *args: Any, **kwargs: Any):
        return TaskFields(Task=self, Time=time, Seed=None)


@dataclass(frozen=True)
class SeedTask(Task, SeedMixin):
    def seed(self, seed: SeedLike):
        return seed_(abs(hash(seed)))

    def log(self, time: float, *args: Any, **kwargs: Any):
        seed = (
            int_(self.seed(s))
            if ((s := kwargs.get("seed", None)) is not None)
            else None
        )
        return TaskFields(Task=self, Time=time, Seed=seed)
