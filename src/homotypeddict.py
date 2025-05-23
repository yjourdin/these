from collections.abc import Iterator
from typing import TYPE_CHECKING, TypedDict


class _HomoTypedDict[T](TypedDict): ...  # type: ignore


# Internal mypy fallback type for all typed dicts (does not exist at runtime)
# N.B. Keep this mostly in sync with typing_extensions._TypedDict/mypy_extensions._TypedDict
if TYPE_CHECKING:
    from _collections_abc import dict_items, dict_keys, dict_values
    from abc import ABCMeta
    from collections.abc import Mapping
    from typing import Any, ClassVar, Never, Self, overload

    class _HomoTypedDict[T](Mapping[str, T], metaclass=ABCMeta):
        __total__: ClassVar[bool]
        __required_keys__: ClassVar[frozenset[str]]
        __optional_keys__: ClassVar[frozenset[str]]
        # __orig_bases__ sometimes exists on <3.12, but not consistently,
        # so we only add it to the stub on 3.12+
        __orig_bases__: ClassVar[tuple[Any, ...]]

        def copy(self) -> Self: ...
        # Using Never so that only calls using mypy plugin hook that specialize the signature
        # can go through.
        def setdefault(self, k: Never, default: T) -> T: ...
        # Mypy plugin hook for 'pop' expects that 'default' has a type variable type.
        def pop(self, k: Never, default: T = ...) -> T: ...  # pyright: ignore[reportInvalidTypeVarUse]
        def update(self, m: Self, /) -> None: ...
        def __delitem__(self, k: Never) -> None: ...
        def items(self) -> dict_items[str, T]: ...
        def keys(self) -> dict_keys[str, T]: ...
        def values(self) -> dict_values[str, T]: ...
        @overload
        def __or__(self, value: Self, /) -> Self: ...  # type: ignore
        @overload
        def __or__(self, value: dict[str, Any], /) -> dict[str, object]: ...
        @overload
        def __ror__(self, value: Self, /) -> Self: ...  # type: ignore
        @overload
        def __ror__(self, value: dict[str, Any], /) -> dict[str, object]: ...
        # supposedly incompatible definitions of __or__ and __ior__
        def __ior__(self, value: Self, /) -> Self: ...  # type: ignore[misc]


class HomoTypedDict[T](_HomoTypedDict[T], TypedDict):  # type: ignore
    def __iter__(self) -> Iterator[str]:
        return super().__iter__()  # type: ignore

    def __len__(self) -> int:
        return super().__len__()  # type: ignore

    def __getitem__(self, key: str) -> T:
        return super().__getitem__(key)  # type: ignore
