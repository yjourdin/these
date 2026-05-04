from abc import ABC
from typing import Any

from .random import Random, RNGParam
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

        original_class.decode = decode
        original_class.encode = encode

        return original_class

    return decorator


class RandomField(Random, Field):
    @staticmethod
    def update_init_dict(init_dict: dict[str, Any], *args: Any, **kwargs: Any):
        return init_dict

    @staticmethod
    def field_random(rng: RNGParam = None, *args: Any, **kwargs: Any) -> Any: ...


def random_field(fieldname: str):
    def decorator[T: type[RandomField]](original_class: T) -> T:
        __class__ = original_class  # noqa: F841

        @classmethod
        def update_init_dict(
            cls: T, init_dict: dict[str, Any], *args: Any, **kwargs: Any
        ) -> dict[str, Any]:
            init_dict = super().update_init_dict(*args, init_dict=init_dict, **kwargs)  # type: ignore
            init_dict[fieldname] = original_class.field_random(*args, **kwargs)
            return init_dict

        original_class.update_init_dict = update_init_dict

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
            super(original_class, original_class).encode(dct)  # type: ignore
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
        def update_init_dict(
            cls: T, init_dict: dict[str, Any] | None = None, *args: Any, **kwargs: Any
        ):
            init_dict = init_dict or {}
            super().update_init_dict(  # type: ignore
                *args, init_dict=init_dict, **kwargs
            )
            init_dict[fieldname] = [
                fieldclass.field_random(*args, **kwargs)
                for _ in range(kwargs["group_size"])
            ]

        original_class.update_init_dict = update_init_dict

        return original_class

    return compose(group_field(fieldname, fieldclass), decorator)


def group_group_field(fieldname: str, fieldclass: type[Field]):
    def decorator[T: type[Field]](original_class: T) -> T:
        @staticmethod
        def decode(dct: dict[Any, Any]):
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

        @staticmethod
        def encode(dct: dict[Any, Any]):
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
