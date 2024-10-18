from dataclasses import dataclass, field
from itertools import count
from typing import ClassVar

from ..dataclass import FrozenDataclass
from ..constants import DEFAULT_MAX_TIME, EPSILON
from ..methods import MethodEnum


@dataclass(frozen=True)
class Config(FrozenDataclass):
    id: int = field(default_factory=count().__next__, init=False)
    method: ClassVar[MethodEnum]
    max_time: int = DEFAULT_MAX_TIME

    def __str__(self) -> str:
        return str(self.id)


@dataclass(frozen=True)
class MIPConfig(Config):
    method = MethodEnum.MIP
    gamma: float = EPSILON


@dataclass(frozen=True)
class SAConfig(Config):
    method = MethodEnum.SA
    accept: float = 0.5
    alpha: float = 0.99
    max_it: int = 20_000


@dataclass(frozen=True)
class SRMPSAConfig(SAConfig):
    amp: float = 0.1


def create_config(**kwargs) -> Config:
    kwargs.pop("id", None)
    match method := kwargs.pop("method", None):
        case MethodEnum.MIP:
            return MIPConfig.from_dict(kwargs)
        case MethodEnum.SA:
            return SRMPSAConfig.from_dict(kwargs)
        case _:
            raise TypeError(f"Unknown method : {method}")
