from dataclasses import asdict, dataclass, fields
from json import dumps, loads
from operator import attrgetter
from typing import Any

from .field import Field, RandomField


@dataclass
class Dataclass(Field):
    @classmethod
    def from_dict(cls, dct: dict[str, Any]):
        return cls(**{
            k: v for k, v in dct.items() if k in [field.name for field in fields(cls)]
        })

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, s: str):
        dct = cls.json_to_dict(s)
        if not dct:
            raise ValueError("Empty json")
        if (classname := dct.pop("__class", cls.__name__)) != cls.__name__:
            raise ValueError(f"Wrong class name : {classname}")
        cls.decode(dct)
        return cls.from_dict(dct)

    def to_json(self):
        dct = self.to_dict()
        super().encode(dct)
        return self.dict_to_json(dct | {"__class": self.__class__.__name__})

    @staticmethod
    def json_to_dict(s: str):
        return loads(s)

    @staticmethod
    def dict_to_json(dct: dict[str, Any]):
        return dumps(dct, indent=4, default=str)

    @staticmethod
    def get_class_name(dct: dict[str, Any]):
        return dct["__class"]

    @staticmethod
    def pop_class_name(dct: dict[str, Any]):
        return dct.pop("__class")


@dataclass(frozen=True)
class FrozenDataclass(Field):
    @classmethod
    def from_dict(cls, dct: dict[str, Any]):
        return cls(**{
            k: v for k, v in dct.items() if k in [field.name for field in fields(cls)]
        })

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, s: str):
        dct = cls.json_to_dict(s)
        if (classname := dct.pop("__class", cls.__name__)) != cls.__name__:
            raise ValueError(f"Wrong class name : {classname}")
        cls.decode(dct)
        return cls.from_dict(dct)

    def to_json(self):
        dct = self.to_dict()
        super().encode(dct)
        return self.dict_to_json(dct | {"__class": self.__class__.__name__})

    @staticmethod
    def json_to_dict(s: str):
        return loads(s)

    @staticmethod
    def dict_to_json(dct: dict[str, Any]):
        return dumps(dct, indent=4, default=str)

    @staticmethod
    def get_class_name(dct: dict[str, Any]):
        return dct["__class"]


@dataclass
class RandomDataclass(Dataclass, RandomField):
    @classmethod
    def random(cls, *args: Any, **kwargs: Any):
        init_dict: dict[Any, Any] = {}
        super().random(init_dict=init_dict, *args, **kwargs)
        return cls(**{
            k: v
            for k, v in init_dict.items()
            if k in map(attrgetter("name"), fields(cls))
        })


@dataclass(frozen=True)
class RandomFrozenDataclass(FrozenDataclass, RandomField):
    @classmethod
    def random(cls, *args: Any, **kwargs: Any):
        init_dict: dict[Any, Any] = {}
        super().random(init_dict=init_dict, *args, **kwargs)
        return cls(**{
            k: v
            for k, v in init_dict.items()
            if k in map(attrgetter("name"), fields(cls))
        })
