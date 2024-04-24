from dataclasses import asdict, dataclass
from json import dumps, loads


@dataclass(frozen=True)
class Config:
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
    pass


@dataclass(frozen=True)
class SAConfig(Config):
    T0_coef: float
    alpha: float
    amp: float
    max_iter: int


CONFIGS = (MIPConfig, SAConfig)
