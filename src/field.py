from abc import ABC, abstractmethod
from typing import Any

from numpy.random import Generator

from .random import Random
from .utils import compose


class Field(ABC):
    @staticmethod
    def decode(dct: dict[Any, Any]) -> dict[Any, Any]:
        return dct

    @staticmethod
    def encode(dct: dict[Any, Any]) -> dict[Any, Any]:
        return dct

    @staticmethod
    def field_decode(o: Any) -> Any:
        return o

    @staticmethod
    def field_encode(o: Any) -> Any:
        return o


def field(fieldname: str):
    def decorator(original_class: Field):
        __class__ = original_class  # noqa: F841

        @staticmethod
        def decode(dct: dict[Any, Any]):
            super().decode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_decode(dct[fieldname])
            return dct

        @staticmethod
        def encode(dct: dict[Any, Any]):
            super().encode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_encode(dct[fieldname])
            return dct

        original_class.decode = decode
        original_class.encode = encode

        return original_class

    return decorator


class RandomField(Random, Field):
    @classmethod
    def random(cls, init_dict: dict[str, Any] = {}, *args: Any, **kwargs: Any): ...

    @staticmethod
    def field_random(rng: Generator, *args: Any, **kwargs: Any) -> Any: ...


def random_field(fieldname: str):
    def decorator(original_class: RandomField):
        __class__ = original_class  # noqa: F841

        @classmethod
        def random(cls, init_dict: dict[str, Any] = {}, *args: Any, **kwargs: Any): # type: ignore
            super().random(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = original_class.field_random(*args, **kwargs)

        original_class.random = random

        return original_class

    return compose(field(fieldname), decorator)


def group_field(fieldname: str, fieldclass: type[Field]):
    def decorator(original_class):
        __class__ = original_class  # noqa: F841

        @classmethod
        def decode(cls, dct: dict):
            super().decode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [fieldclass.field_decode(o) for o in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )
            return dct

        @classmethod
        def encode(cls, dct: dict):
            super().encode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [fieldclass.field_encode(o) for o in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )
            return dct

        original_class.decode = decode
        original_class.encode = encode

        return original_class

    return decorator


def random_group_field(fieldname: str, fieldclass: type[RandomField]):
    def decorator(original_class):
        __class__ = original_class  # noqa: F841

        @classmethod
        def random(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().random(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = [
                fieldclass.field_random(*args, **kwargs)
                for _ in range(kwargs["group_size"])
            ]

        original_class.random = random

        return original_class

    return compose(group_field(fieldname, fieldclass), decorator)


def group_group_field(fieldname: str, fieldclass: type[Field]):
    def decorator(original_class):
        __class__ = original_class  # noqa: F841

        @classmethod
        def decode(cls, dct: dict):
            super().decode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [[fieldclass.field_decode(o) for o in l] for l in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )
            return dct

        @classmethod
        def encode(cls, dct: dict):
            super().encode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [[fieldclass.field_encode(o) for o in l] for l in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )
            return dct

        original_class.decode = decode
        original_class.encode = encode

        return original_class

    return decorator
