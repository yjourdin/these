from abc import ABC


class Field(ABC):
    @classmethod
    def json_to_dict(cls, dct: dict):
        return dct

    @classmethod
    def dict_to_json(cls, dct: dict):
        return dct


class GeneratedField(Field):
    @classmethod
    def random(cls, *args, **kwargs): ...

    @classmethod
    def balanced(cls, *args, **kwargs): ...


def group_field(fieldname: str, fieldclass: type[Field]):
    def decorator(cls):
        __class__ = cls  # noqa: F841

        @classmethod
        def json_to_dict(cls, dct: dict):
            super().json_to_dict(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.json_to_dict({fieldname: o}) for o in dct[fieldname]
                ]

        @classmethod
        def dict_to_json(cls, dct: dict):
            super().dict_to_json(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.dict_to_json({fieldname: o}) for o in dct[fieldname]
                ]

        cls.json_to_dict = json_to_dict
        cls.dict_to_json = dict_to_json

        return cls

    return decorator


def group_generated_field(fieldname: str, fieldclass: type[Field]):
    def decorator(cls):
        __class__ = cls  # noqa: F841

        @classmethod
        def json_to_dict(cls, dct: dict):
            super().json_to_dict(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.json_to_dict({fieldname: o}) for o in dct[fieldname]
                ]

        @classmethod
        def dict_to_json(cls, dct: dict):
            super().dict_to_json(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.dict_to_json({fieldname: o}) for o in dct[fieldname]
                ]

        @classmethod
        def random(cls, *args, **kwargs):
            super().random(*args, **kwargs)  # type: ignore
            kwargs[fieldname] = [
                fieldclass.random(*args, **kwargs)  # type: ignore
                for _ in kwargs["size"]
            ]

        @classmethod
        def balanced(cls, *args, **kwargs):
            super().balanced(*args, **kwargs)  # type: ignore
            kwargs[fieldname] = [
                fieldclass.balanced(*args, **kwargs)  # type: ignore
                for _ in kwargs["size"]
            ]

        cls.json_to_dict = json_to_dict
        cls.dict_to_json = dict_to_json
        cls.random = random
        cls.balanced = balanced

        return cls

    return decorator
