from dataclasses import asdict, dataclass, field
from json import dumps, loads

from numpy.random import Generator


@dataclass
class Seeds:
    A_train: list[int] = field(default_factory=list)
    A_test: list[int] = field(default_factory=list)
    Mo: list[int] = field(default_factory=list)

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


def seed(rng: Generator) -> int:
    return rng.integers(2**63)


def seeds(rng: Generator, nb: int = 1) -> list[int]:
    return rng.integers(2**63, size=nb).tolist()
