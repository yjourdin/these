from abc import ABC
from typing import Any


class Field(ABC):
    @classmethod
    def json_to_dict(cls, dct: dict):
        return dct

    @classmethod
    def dict_to_json(cls, dct: dict):
        return dct

    @staticmethod
    def field_to_dict(o):
        return o

    @staticmethod
    def field_to_json(o):
        return o


def field(fieldname: str):
    def decorator(original_class):
        __class__ = original_class  # noqa: F841

        @classmethod
        def json_to_dict(cls, dct: dict):
            super().json_to_dict(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_to_dict(dct[fieldname])
                return dct[fieldname]

        @classmethod
        def dict_to_json(cls, dct: dict):
            super().dict_to_json(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_to_json(dct[fieldname])
                return dct[fieldname]

        original_class.json_to_dict = json_to_dict
        original_class.dict_to_json = dict_to_json

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
        def json_to_dict(cls, dct: dict):
            super().json_to_dict(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_to_dict(dct[fieldname])
                return dct[fieldname]

        @classmethod
        def dict_to_json(cls, dct: dict):
            super().dict_to_json(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = original_class.field_to_json(dct[fieldname])
                return dct[fieldname]

        @classmethod
        def random(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().random(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = original_class.field_random(*args, **kwargs)

        @classmethod
        def balanced(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().balanced(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = original_class.field_balanced(*args, **kwargs)

        original_class.json_to_dict = json_to_dict
        original_class.dict_to_json = dict_to_json
        original_class.random = random
        original_class.balanced = balanced

        return original_class

    return decorator


def group_field(fieldname: str, fieldclass: type[Field]):
    def decorator(original_class):
        __class__ = original_class  # noqa: F841

        @classmethod
        def json_to_dict(cls, dct: dict):
            super().json_to_dict(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [fieldclass.field_to_dict(o) for o in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )

        @classmethod
        def dict_to_json(cls, dct: dict):
            super().dict_to_json(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [fieldclass.field_to_json(o) for o in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )

        original_class.json_to_dict = json_to_dict
        original_class.dict_to_json = dict_to_json

        return original_class

    return decorator


def group_generated_field(fieldname: str, fieldclass: type[GeneratedField]):
    def decorator(original_class):
        __class__ = original_class  # noqa: F841

        @classmethod
        def json_to_dict(cls, dct: dict):
            super().json_to_dict(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [fieldclass.field_to_dict(o) for o in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )

        @classmethod
        def dict_to_json(cls, dct: dict):
            super().dict_to_json(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = (
                    [fieldclass.field_to_json(o) for o in dct[fieldname]]
                    if dct[fieldname]
                    else None
                )

        @classmethod
        def random(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().random(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = [
                fieldclass.field_random(*args, **kwargs) for _ in range(kwargs["size"])
            ]

        @classmethod
        def balanced(cls, init_dict: dict[str, Any] = {}, *args, **kwargs):
            super().balanced(init_dict=init_dict, *args, **kwargs)  # type: ignore
            init_dict[fieldname] = [
                fieldclass.field_balanced(*args, **kwargs)
                for _ in range(kwargs["size"])
            ]

        original_class.json_to_dict = json_to_dict
        original_class.dict_to_json = dict_to_json
        original_class.random = random
        original_class.balanced = balanced

        return original_class

    return decorator
