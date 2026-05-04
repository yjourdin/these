from enum import Enum, member
from typing import Any

from .function import generate_linext, generate_weak_order, generate_weak_order_ext


class ScriptEnum(Enum):
    @member
    def LINEXT(self, *args: Any, **kwargs: Any):
        return generate_linext(*args, **kwargs)

    @member
    def WEAK_ORDER(self, *args: Any, **kwargs: Any):
        return generate_weak_order(*args, **kwargs)

    @member
    def WEAK_ORDER_EXT(self, *args: Any, **kwargs: Any):
        return generate_weak_order_ext(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any):
        return self.value(self, *args, **kwargs)
