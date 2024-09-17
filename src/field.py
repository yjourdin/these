from abc import ABC
from typing import Any


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


class GeneratedField(Field):
    @classmethod
    def random(cls, *args, **kwargs): ...

    @classmethod
    def balanced(cls, *args, **kwargs): ...

    @staticmethod
    def field_random(*args, **kwargs): ...

    @staticmethod
    def field_balanced(*args, **kwargs): ...


def generated_field(fieldname: str):
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

        @classmethod
        def random(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().random(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = original_class.field_random(*args, **kwargs)

        @classmethod
        def balanced(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().balanced(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = original_class.field_balanced(*args, **kwargs)

        original_class.decode = decode
        original_class.encode = encode
        original_class.random = random
        original_class.balanced = balanced

        return original_class

    return decorator


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


def group_generated_field(fieldname: str, fieldclass: type[GeneratedField]):
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

        @classmethod
        def random(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().random(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = [
                fieldclass.field_random(*args, **kwargs)
                for _ in range(kwargs["group_size"])
            ]

        @classmethod
        def balanced(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().balanced(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = [
                fieldclass.field_balanced(*args, **kwargs)
                for _ in range(kwargs["group_size"])
            ]

        original_class.decode = decode
        original_class.encode = encode
        original_class.random = random
        original_class.balanced = balanced

        return original_class

    return decorator
