from abc import abstractmethod
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_EXCEPTION, Future, as_completed, wait
from dataclasses import dataclass, fields
from time import process_time
from typing import Any, ClassVar, NamedTuple, TypeGuard

from ..dataclass import FrozenDataclass
from ..random import Seed, SeedMixin
from .directory import Directory
from .fieldnames import SeedFieldnames, TimeFieldnames


class TaskResult(NamedTuple):
    result: Any
    time: float


type FutureTask = Future[TaskResult]
type FutureTaskException = Future[TaskResult | None]


def wait_exception(future: FutureTaskException) -> TypeGuard[FutureTask]:
    if (err := future.exception()) is not None:
        raise err
    return True


def wait_exception_iterable(
    futures: Iterable[FutureTaskException],
) -> TypeGuard[Iterable[FutureTask]]:
    for future in as_completed(futures):
        wait_exception(future)
    return True


def wait_exception_mapping(
    futures: Mapping[Any, FutureTaskException],
) -> TypeGuard[Mapping[Any, FutureTask]]:
    return wait_exception_iterable(futures.values())


@dataclass(frozen=True)
class Task(FrozenDataclass):
    name: ClassVar[str]

    def __str__(self) -> str:
        return f"{self.name:10} ({', '.join(f'{field.name}: {str(getattr(self, field.name))}' for field in fields(self))})"

    def __call__(self, dir: Directory, *args, **kwargs):
        tic = process_time()
        result = self.task(dir=dir, *args, **kwargs)
        toc = process_time()

        time = toc - tic
        dir.csv_files["times"].queue.put(
            {TimeFieldnames.Task: self, TimeFieldnames.Time: time}
        )

        return TaskResult(result, time)

    @abstractmethod
    def task(self, dir: Directory, *args, **kwargs): ...

    @abstractmethod
    def done(self, *args, **kwargs) -> bool: ...


@dataclass(frozen=True)
class SeedTask(Task, SeedMixin):
    def seed(self, seed: Seed):
        return abs(hash((self, seed)))

    def __call__(self, dir: Directory, *args, **kwargs):
        if "seed" in kwargs:
            dir.csv_files["seeds"].queue.put(
                {
                    SeedFieldnames.Task: self,
                    SeedFieldnames.Seed: self.seed(kwargs["seed"]),
                }
            )
        return super().__call__(dir=dir, *args, **kwargs)
