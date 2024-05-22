from dataclasses import asdict, dataclass, field
from json import dumps, loads
from typing import Literal

from jobs import JOBS
from .seed import Seeds

from .config import CONFIGS, Config

Model = Literal["RMP", "SRMP"]
Method = Literal["MIP", "SA"]
ConfigDict = dict[Method, dict[int, Config]]


def config_hook(dct):
    for config in CONFIGS:
        try:
            return config.from_dict(dct)
        except TypeError:
            pass
    return dct


@dataclass
class Arguments:
    name: str
    jobs: int = JOBS
    seed: int | None = None
    nb_A_tr: int = 1
    nb_A_te: int = 1
    nb_Mo: int = 1
    seeds: Seeds = Seeds()
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
    config: ConfigDict = field(default_factory=dict)

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
