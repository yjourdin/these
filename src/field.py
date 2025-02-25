from abc import ABC
from typing import Any

from numpy.random import Generator

from .random import Random
from .utils import compose


class Field(ABC):
    @classmethod
    def decode(cls, dct: dict[Any, Any]) -> dict[Any, Any]:
        return dct

    @classmethod
    def encode(cls, dct: dict[Any, Any]) -> dict[Any, Any]:
        return dct

    @staticmethod
    def field_decode(o: Any) -> Any:
        return o

    @staticmethod
    def field_encode(o: Any) -> Any:
        return o


def field(fieldname: str):
    def decorator[T: type[Field]](original_class: T) -> T:
        __class__ = original_class  # noqa: F841

        @classmethod
        def decode(cls: T, dct: dict[Any, Any]):
            super().decode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_decode(dct[fieldname])
            return dct

        @classmethod
        def encode(cls: T, dct: dict[Any, Any]):
            super().encode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_encode(dct[fieldname])
            return dct

        original_class.decode = classmethod(decode)
        original_class.encode = encode

        return original_class

    return decorator


class RandomField(Random, Field):
    @classmethod
    def random(
        cls, init_dict: dict[str, Any] = {}, *args: Any, **kwargs: Any
    ) -> "RandomField": ...

    @staticmethod
    def field_random(rng: Generator, *args: Any, **kwargs: Any) -> Any: ...


def random_field(fieldname: str):
    def decorator[T: type[RandomField]](original_class: T) -> T:
        __class__ = original_class  # noqa: F841

        @classmethod
        def random(cls: T, init_dict: dict[str, Any] = {}, *args: Any, **kwargs: Any):
            super().random(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = original_class.field_random(*args, **kwargs)

        original_class.random = random

        return original_class

    return compose(field(fieldname), decorator)


def group_field(fieldname: str, fieldclass: type[Field]):
    def decorator[T: type[Field]](original_class: T) -> T:
        __class__ = original_class  # noqa: F841

        @classmethod
        def decode(cls: T, dct: dict[Any, Any]):
            super().decode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [fieldclass.field_decode(o) for o in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )
            return dct

        @classmethod
        def encode(cls: T, dct: dict[Any, Any]):
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
    def decorator[T: type[RandomField]](original_class: T) -> T:
        __class__ = original_class  # noqa: F841

        @classmethod
        def random(cls: T, init_dict: dict[str, Any] = {}, *args: Any, **kwargs: Any):
            super().random(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = [
                fieldclass.field_random(*args, **kwargs)
                for _ in range(kwargs["group_size"])
            ]

        original_class.random = random  # type: ignore

        return original_class

    return compose(group_field(fieldname, fieldclass), decorator)


def group_group_field(fieldname: str, fieldclass: type[Field]):
    def decorator[T: type[Field]](original_class: T) -> T:
        __class__ = original_class  # noqa: F841

        @classmethod
        def decode(cls: T, dct: dict[Any, Any]):
            super().decode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [
                        [fieldclass.field_decode(o) for o in lst]
                        for lst in dct[fieldname]
                    ]
                    if dct[fieldname]
                    else None
                )
            return dct

        @classmethod
        def encode(cls: T, dct: dict[Any, Any]):
            super().encode(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [
                        [fieldclass.field_encode(o) for o in lst]
                        for lst in dct[fieldname]
                    ]
                    if dct[fieldname]
                    else None
                )
            return dct

        original_class.decode = decode
        original_class.encode = encode

        return original_class

    return decorator
