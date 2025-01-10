from time import process_time
from abc import abstractmethod
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from ..dataclass import FrozenDataclass
from ..random import Seed, SeedMixin
from .directory import Directory
from .fieldnames import SeedFieldnames, TimeFieldnames


@dataclass(frozen=True)
class Task(FrozenDataclass):
    name: ClassVar[str]

    def __str__(self) -> str:
        return f"{self.name:10} ({', '.join(f"{field.name}: {str(getattr(self, field.name))}" for field in fields(self))})"

    def __call__(self, dir: Directory, *args, **kwargs) -> Any:
        tic = process_time()
        result = self.task(dir=dir, *args, **kwargs)
        toc = process_time()
        dir.csv_files["times"].queue.put(
            {TimeFieldnames.Task: self, TimeFieldnames.Time: toc - tic}
        )
        return result

    @abstractmethod
    def task(self, dir: Directory, *args, **kwargs) -> Any: ...

    @abstractmethod
    def done(self, *args, **kwargs) -> bool: ...


@dataclass(frozen=True)
class SeedTask(Task, SeedMixin):
    def seed(self, seed: Seed):
        return abs(hash((self, seed)))

    def __call__(self, dir: Directory, *args, **kwargs) -> Any:
        if "seed" in kwargs:
            dir.csv_files["seeds"].queue.put(
                {
                    SeedFieldnames.Task: self,
                    SeedFieldnames.Seed: self.seed(kwargs["seed"]),
                }
            )
        return super().__call__(dir=dir, *args, **kwargs)
