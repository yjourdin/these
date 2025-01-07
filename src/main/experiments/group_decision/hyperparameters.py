from dataclasses import dataclass, field
from itertools import count
from typing import ClassVar

from src.enum_base import StrEnum

from ....dataclass import FrozenDataclass


class TypeEnum(StrEnum):
    GEN = "Gen"
    ACCEPT = "Accept"


@dataclass(frozen=True)
class Hyperparameters(FrozenDataclass):
    id: int = field(default_factory=count().__next__, init=False, hash=False)
    type: ClassVar[TypeEnum]

    def __str__(self) -> str:
        return str(self.id)


@dataclass(frozen=True)
class GenHyperparameters(Hyperparameters):
    type = TypeEnum.GEN
    P: float
    W: float
    L: int


@dataclass(frozen=True)
class AcceptHyperparameters(Hyperparameters):
    type = TypeEnum.ACCEPT
    P: float
    W: float
    L: int
