from dataclasses import asdict, dataclass
from json import dumps, loads
from typing import ClassVar

from .types import Method


@dataclass(frozen=True)
class Config:
    id: int
    method: ClassVar[Method]

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(loads(s))

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return dumps(self.to_dict(), indent=4)


@dataclass(frozen=True)
class MIPConfig(Config):
    method = "MIP"


@dataclass(frozen=True)
class SAConfig(Config):
    method = "SA"
    T0_coef: float
    alpha: float
    amp: float
    max_iter: int


def create_config(dct: dict) -> Config | dict:
    id = dct.get("id", None)
    if id is not None:
        method = dct.pop("method", None)
        match method:
            case "MIP":
                return MIPConfig.from_dict(dct)
            case "SA":
                return SAConfig.from_dict(dct)
            case _:
                raise ValueError("Unknown method")
    else:
        return dct
