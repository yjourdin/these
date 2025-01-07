from abc import ABC
from typing import Any

from numpy.random import Generator

from .random import Random
from .utils import compose


class Field(ABC):
    @classmethod
    def decode(cls, dct: dict):
        return dct

    @classmethod
    def encode(cls, dct: dict):
        return dct

    @staticmethod
    def field_decode(o):
        return o

    @staticmethod
    def field_encode(o):
        return o


def field(fieldname: str):
    def decorator(original_class):
        __class__ = original_class  # noqa: F841

        @classmethod
        def decode(cls, dct: dict):
            super().decode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_decode(dct[fieldname])
            return dct

        @classmethod
        def encode(cls, dct: dict):
            super().encode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_encode(dct[fieldname])
            return dct

        original_class.decode = decode
        original_class.encode = encode

        return original_class

    return decorator


class RandomField(Random, Field):
    @staticmethod
    def field_random(rng: Generator, *args, **kwargs): ...


def random_field(fieldname: str):
    def decorator(original_class):
        __class__ = original_class  # noqa: F841

        @classmethod
        def random(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
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