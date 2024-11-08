from enum import Enum as BaseEnum


class Enum(BaseEnum):
    def __str__(self) -> str:
        return self.name


class StrEnum(str, BaseEnum):
    def __str__(self) -> str:
        return self.value
