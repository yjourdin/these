from enum import auto

from .case_insensitive_str_enum import CaseInsensitiveStrEnum


class MethodEnum(CaseInsensitiveStrEnum):
    MIP = auto()
    SA = auto()
