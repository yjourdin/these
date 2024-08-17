from dataclasses import asdict, dataclass, fields
from json import dumps, loads

from .field import Field, GeneratedField


@dataclass(kw_only=True)
class Dataclass(Field):
    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    def to_dict(self):
        return asdict(self)

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
        return asdict(self)

    @classmethod
    def from_json(cls, s):
        dct = loads(s)
        super().json_to_dict(dct)
        return cls.from_dict(dct)

    def to_json(self):
        dct = self.to_dict()
        super().dict_to_json(dct)
        return dumps(dct, indent=4)


@dataclass
class GeneratedDataclass(Dataclass, GeneratedField):
    @classmethod
    def random(cls, *args, **kwargs):
        init_dict = kwargs
        super().random(init_dict=init_dict, *args, **kwargs)
        return cls(**{k.name: kwargs[k.name] for k in fields(cls)})

    @classmethod
    def balanced(cls, *args, **kwargs):
        init_dict = kwargs
        super().random(init_dict=init_dict, *args, **kwargs)
        return cls(**{k.name: kwargs[k.name] for k in fields(cls)})
