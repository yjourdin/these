from enum import Flag, StrEnum  # type: ignore
from typing import Any


class StrEnum(StrEnum):
    @classmethod
    def _missing_(cls, value: Any):
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member.value == value:
                    return member
        return None


class ParamFlag(Flag): ...
