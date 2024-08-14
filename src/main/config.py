from dataclasses import dataclass, field
from itertools import count
from typing import ClassVar

from ..dataclass import FrozenDataclass
from ..methods import MethodEnum


@dataclass(frozen=True)
class Config(FrozenDataclass):
    id: int = field(default_factory=count().__next__, init=False)
    method: ClassVar[MethodEnum]


@dataclass(frozen=True)
class MIPConfig(Config):
    method = MethodEnum.MIP
    gamma: float = 0.001


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
            return SAConfig.from_dict(kwargs)
        case _:
            raise TypeError(f"Unknown method : {method}")
