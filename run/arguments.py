from dataclasses import asdict, dataclass, field
from json import dumps, loads
from typing import Literal

from .config import SAConfig

JOBS = 70

Model = Literal["RMP", "SRMP"]
Method = Literal["MIP", "SA"]


def config_hook(dct):
    """Transform dict keys to int if possible

    :param dct: Dict to modify
    :return : Modified dict
    """
    try:
        return SAConfig.from_dict(dct)
    except TypeError:
        return dct


@dataclass
class Arguments:
    name: str
    jobs: int = JOBS
    seed: int | None = None
    seeds: list[int] | int = 1
    N_tr: list[int] = field(default_factory=list)
    N_te: list[int] = field(default_factory=list)
    method: list[Method] = field(default_factory=list)
    M: list[int] = field(default_factory=list)
    Mo: list[Model] = field(default_factory=list)
    Ko: list[int] = field(default_factory=list)
    N_bc: list[int] = field(default_factory=list)
    Me: list[Model] = field(default_factory=list)
    Ke: list[int] = field(default_factory=list)
    error: list[float] = field(default_factory=list)
    config: dict[int, SAConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(loads(s, object_hook=config_hook))

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return dumps(self.to_dict(), indent=4)
