from dataclasses import asdict, dataclass, fields
from json import dumps, loads
from operator import attrgetter

from .field import Field, GeneratedField


@dataclass
class Dataclass(Field):
    @classmethod
    def decode(cls, dct: dict):
        dct = super().decode(dct)
        return dct

    @classmethod
    def encode(cls, dct: dict):
        dct = super().encode(dct)
        return dct

    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, s: str):
        dct = cls.json_to_dict(s)
        if not dct:
            raise ValueError("Empty json")
        classname = dct.pop("__class", cls.__name__)
        if classname != cls.__name__:
            raise ValueError(f"Wrong class name : {classname}")
        cls.decode(dct)
        return cls.from_dict(dct)

    def to_json(self):
        dct = self.to_dict()
        type(self).encode(dct)
        return self.dict_to_json(dct)

    @staticmethod
    def json_to_dict(s: str):
        return loads(s)

    @classmethod
    def dict_to_json(cls, dct: dict):
        return dumps(dct | {"__class": cls.__name__}, indent=4, default=str)

    @staticmethod
    def get_class_name(dct: dict):
        return dct["__class"]

    @staticmethod
    def pop_class_name(dct: dict):
        return dct.pop("__class")


@dataclass(frozen=True)
class FrozenDataclass(Field):
    @classmethod
    def from_dict(cls, dct: dict):
        return cls(**dct)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, s: str):
        dct = cls.json_to_dict(s)
        classname = dct.pop("__class", cls.__name__)
        if classname != cls.__name__:
            raise ValueError(f"Wrong class name : {classname}")
        super().decode(dct)
        return cls.from_dict(dct)

    def to_json(self):
        dct = self.to_dict()
        super().encode(dct)
        return self.dict_to_json(dct)

    @staticmethod
    def json_to_dict(s: str):
        return loads(s)

    @classmethod
    def dict_to_json(cls, dct: dict):
        return dumps(dct | {"__class": cls.__name__}, indent=4)

    @staticmethod
    def get_class_name(dct: dict):
        return dct["__class"]


@dataclass
class GeneratedDataclass(Dataclass, GeneratedField):
    @classmethod
    def random(cls, *args, **kwargs):
        init_dict = {}
        super().random(init_dict=init_dict, *args, **kwargs)
        return cls(
            **{
                k: v
                for k, v in init_dict.items()
                if k in map(attrgetter("name"), fields(cls))
            }
        )

    @classmethod
    def balanced(cls, *args, **kwargs):
        init_dict = {}
        super().balanced(init_dict=init_dict, *args, **kwargs)
        return cls(
            **{
                k: v
                for k, v in init_dict.items()
                if k in map(attrgetter("name"), fields(cls))
            }
        )
