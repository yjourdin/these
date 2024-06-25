from abc import ABC
from dataclasses import asdict, dataclass, field
from itertools import count
from json import dumps, loads
from typing import ClassVar

from .types import Method


@dataclass(frozen=True)
class Config(ABC):
    id: int = field(default_factory=count().__next__, init=False)
    method: ClassVar[Method]

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(loads(s))

    def to_dict(self):
        return asdict(self) | {"method": self.method}

    def to_json(self):
        return dumps(self.to_dict(), indent=4)


@dataclass(frozen=True)
class MIPConfig(Config):
    method = "MIP"
    gamma: float = 0.001


@dataclass(frozen=True)
class SAConfig(Config):
    method = "SA"
    T0_coef: float = 1
    alpha: float = 0.9999
    amp: float = 0.1
    max_iter: int = 20_000


def create_config(**kwargs) -> Config:
    kwargs.pop("id", None)
    method = kwargs.pop("method", None)
    match method:
        case "MIP":
            return MIPConfig.from_dict(kwargs)
        case "SA":
            return SAConfig.from_dict(kwargs)
        case _:
            raise TypeError(f"Unknown method : {method}")
