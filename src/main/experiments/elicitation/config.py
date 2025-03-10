from dataclasses import dataclass, field
from itertools import count
from typing import Any, ClassVar

from ....constants import DEFAULT_MAX_TIME, EPSILON
from ....dataclass import FrozenDataclass
from ....methods import MethodEnum


@dataclass(frozen=True)
class Config(FrozenDataclass):
    id: int = field(default_factory=count().__next__, init=False, hash=False)
    method: ClassVar[MethodEnum]
    max_time: int = field(default=DEFAULT_MAX_TIME, hash=False)

    def __str__(self) -> str:
        return str(self.id)


@dataclass(frozen=True)
class MIPConfig(Config):
    method = MethodEnum.MIP
    gamma: float = EPSILON


@dataclass(frozen=True)
class SAConfig(Config):
    method = MethodEnum.SA
    accept: float = 0.9
    alpha: float = 0.999
    amp: float = 0.05
    max_it: int | None = None
    max_it_non_improving: int | None = None


def create_config(**kwargs: Any) -> Config:
    kwargs.pop("id", None)
    if kwargs.get("method"):
        method = kwargs["method"]
    else:
        method = eval(Config.get_class_name(kwargs)).method
    if not isinstance(method, str):
        raise TypeError(f"Unknown method : {method}")
    match method.lower():
        case MethodEnum.MIP:
            return MIPConfig.from_dict(kwargs)
        case MethodEnum.SA:
            return SAConfig.from_dict(kwargs)
        case _:
            raise TypeError(f"Unknown method : {method}")
