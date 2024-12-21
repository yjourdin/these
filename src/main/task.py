from abc import abstractmethod
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from ..dataclass import FrozenDataclass
from ..random import Seed, SeedMixin
from .directory import Directory


@dataclass(frozen=True)
class Task(FrozenDataclass):
    name: ClassVar[str]

    def __str__(self) -> str:
        return f"{self.name:10} ({', '.join(f"{field.name}: {str(getattr(self, field.name)):3}" for field in fields(self))})"

    @abstractmethod
    def __call__(self, dir: Directory, *args, **kwargs) -> Any: ...

    @abstractmethod
    def done(self, *args, **kwargs) -> bool: ...


@dataclass(frozen=True)
class SeedTask(Task, SeedMixin):
    def seed(self, seed: Seed):
        return abs(hash((self, seed)))
