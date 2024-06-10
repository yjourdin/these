from dataclasses import asdict, dataclass, field
from json import dumps, loads

from jobs import JOBS

from .config import Config, create_config
from .seed import Seeds
from .types import Method, Model


@dataclass
class Arguments:
    name: str
    jobs: int = JOBS
    seed: int | None = None
    nb_A_tr: int = 1
    nb_Mo: int | None = None
    nb_A_te: int | None = None
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
    config: list[Config] = field(default_factory=list)

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(loads(s, object_hook=create_config))

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return dumps(self.to_dict(), indent=4)
