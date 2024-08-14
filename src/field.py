from abc import ABC, ABCMeta


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
        def json_to_dict(cls, dct: dict):
            super(type(cls), cls).json_to_dict(dct)
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.json_to_dict({fieldname: o}) for o in dct[fieldname]
                ]

        def dict_to_json(cls, dct: dict):
            super(type(cls), cls).dict_to_json(dct)
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.dict_to_json({fieldname: o}) for o in dct[fieldname]
                ]

        cls.json_to_dict = json_to_dict
        cls.dict_to_json = dict_to_json

        return cls
    return decorator


class GroupField(ABCMeta):
    def __new__(metacls, clsname, bases, namespace, fieldname, fieldclass):
        def json_to_dict(cls, dct: dict):
            super(clsname, cls).json_to_dict(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.json_to_dict({fieldname: o}) for o in dct[fieldname]
                ]

        def dict_to_json(cls, dct: dict):
            super(clsname, cls).dict_to_json(dct)  # type: ignore
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.dict_to_json({fieldname: o}) for o in dct[fieldname]
                ]

        return super().__new__(
            metacls,
            clsname,
            bases,
            namespace | {"json_to_dict": json_to_dict, "dict_to_json": dict_to_json},
        )
        
        
def group_generated_field(fieldname: str, fieldclass: type[Field]):
    def decorator(cls):
        def json_to_dict(cls, dct: dict):
            super(type(cls), cls).json_to_dict(dct)
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.json_to_dict({fieldname: o}) for o in dct[fieldname]
                ]

        def dict_to_json(cls, dct: dict):
            super(type(cls), cls).dict_to_json(dct)
            if fieldname in dct:
                dct[fieldname] = [
                    fieldclass.dict_to_json({fieldname: o}) for o in dct[fieldname]
                ]
        
        def random(cls, *args, **kwargs):
            super(type(cls), cls).random(*args, **kwargs)
            kwargs[fieldname] = [
                fieldclass.random(*args, **kwargs) for _ in kwargs["size"]
            ]

        def balanced(cls, *args, **kwargs):
            super(type(cls), cls).balanced(*args, **kwargs)
            kwargs[fieldname] = [
                fieldclass.balanced(*args, **kwargs) for _ in kwargs["size"]
            ]

        cls.json_to_dict = json_to_dict
        cls.dict_to_json = dict_to_json
        cls.random = random
        cls.balanced = balanced

        return cls
    return decorator


class GroupGeneratedField(GroupField):
    def __new__(metacls, clsname, bases, namespace, fieldname, fieldclass):
        def random(cls, *args, **kwargs):
            super(clsname, cls).random(*args, **kwargs)
            kwargs[fieldname] = [
                fieldclass.random(*args, **kwargs) for _ in kwargs["size"]
            ]

        def balanced(cls, *args, **kwargs):
            super(clsname, cls).balanced(*args, **kwargs)
            kwargs[fieldname] = [
                fieldclass.balanced(*args, **kwargs) for _ in kwargs["size"]
            ]

        return super().__new__(
            metacls,
            clsname,
            bases,
            namespace | {"random": random, "balanced": balanced},
            fieldname=fieldname,
            fieldclass=fieldclass,
        )
