from dataclasses import asdict, dataclass
from json import dumps, loads


def key2int(dct):
    """Transform dict keys to int if possible

    :param dct: Dict to modify
    :return : Modified dict
    """
    try:
        return {int(k): v for k, v in dct.items()}
    except ValueError:
        return dct


@dataclass(frozen=True)
class SAConfig:
    T0: dict[int, float]
    alpha: float
    amp: float
    max_iter: int

    @classmethod
    def from_dict(cls, dct):
        return cls(
            **{
                k: key2int(v) if isinstance(v, dict) else v for k, v in dct.items()
            }  # type: ignore
        )

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(loads(s, object_hook=key2int))

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return dumps(self.to_dict(), indent=4)
