from dataclasses import asdict, dataclass, fields
from json import dumps, loads
from typing import cast

from typing_extensions import Self

from .field import Field, GeneratedField
from .utils import compose


@dataclass(kw_only=True)
class Dataclass(Field):
    @classmethod
    def from_dict(cls, dct: dict):
        c = cast(type[Self], eval(dct.pop("dataclass", cls.__name__)))
        return c(**dct)

    def to_dict(self):
        return asdict(self) | {"dataclass": self.__class__.__name__}

    @classmethod
    def from_json(cls, s):
        dct = loads(s)
        super().json_to_dict(dct)
        return cls.from_dict(dct)

    def to_json(self):
        dct = self.to_dict()
        super().dict_to_json(dct)
        return dumps(dct, indent=4)


@dataclass(frozen=True)
class FrozenDataclass(Field):
    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    def to_dict(self):
        return asdict(self) | {"dataclass": self.__class__.__name__}

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(
            loads(
                s,
                object_hook=compose(*cls.json_to_dict),  # type: ignore
            )
        )

    def to_json(self):
        return dumps(
            self.to_dict(),
            indent=4,
            default=compose(*self.dict_to_json),  # type: ignore
        )


@dataclass
class GeneratedDataclass(Dataclass, GeneratedField):
    @classmethod
    def random(cls, *args, **kwargs):
        super().random(*args, **kwargs)
        return cls(**{k.name: kwargs[k.name] for k in fields(cls)})

    @classmethod
    def balanced(cls, *args, **kwargs):
        super().random(*args, **kwargs)
        return cls(**{k.name: kwargs[k.name] for k in fields(cls)})
