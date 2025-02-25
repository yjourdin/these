from abc import abstractmethod
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, as_completed
from dataclasses import dataclass, fields
from time import process_time
from typing import Any, ClassVar, NamedTuple, TypeGuard

from ..dataclass import FrozenDataclass
from ..random import Seed, SeedMixin
from .abstract_task import AbstractTask
from .csv_files import TaskFields
from .directory import Directory


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
class Task(FrozenDataclass, AbstractTask):
    name: ClassVar[str]

    def __str__(self) -> str:
        return f"{self.name:10} ({', '.join(f'{field.name}: {str(getattr(self, field.name))}' for field in fields(self))})"

    def __call__(self, dir: Directory, *args: Any, **kwargs: Any):
        tic = process_time()
        result = self.task(dir=dir, *args, **kwargs)
        toc = process_time()

        time = toc - tic
        csv_file = dir.csv_files["tasks"]
        csv_file.writerow(self.log(csv_file.fields, time, *args, **kwargs))

        return TaskResult(result, time)

    @abstractmethod
    def task(self, dir: Directory, *args: Any, **kwargs: Any) -> Any: ...

    @abstractmethod
    def done(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def log(self, fields: type[TaskFields], time: float, *args: Any, **kwargs: Any):
        return fields(Task=self, Time=time, Seed=None)


@dataclass(frozen=True)
class SeedTask(Task, SeedMixin):
    def seed(self, seed: Seed) -> Seed:
        return abs(hash((self, seed)))

    def log(self, fields: type[TaskFields], time: float, *args: Any, **kwargs: Any):
        seed = self.seed(s) if ((s := kwargs.get("seed", None)) is not None) else None
        return fields(Task=self, Time=time, Seed=seed)
