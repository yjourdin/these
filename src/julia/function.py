import ast
from itertools import chain
from subprocess import run
from typing import Any

from ..utils import int_to_ind_set
from .file import PARENT_DIR, S_file, WE_file


def run_julia(scriptname: str, *args: Any, **kwargs: Any):
    return run(
        ["julia", "--project", PARENT_DIR / scriptname]
        + [str(x) for x in args]
        + list(
            chain.from_iterable(
                (f"--{k}", str(v)) for k, v in kwargs.items() if v is not None
            )
        ),
        capture_output=True,
        text=True,
    ).stdout


def python_exec(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        raise Exception(f"Julia output : {s}")


def generate_linext(m: int, seed: int | None = None) -> list[list[int]]:
    linext = python_exec(run_julia("generate_linext.jl", m, seed=seed))

    return [int_to_ind_set(i) for i in linext]


def generate_partial_sum(m: int) -> None:
    return python_exec(run_julia("generate_partial_sum.jl", m, output=S_file(m)))


def generate_weak_order(m: int, seed: int | None = None) -> list[int]:
    file = S_file(m)

    if not file.exists():
        generate_partial_sum(m)

    return python_exec(run_julia("generate_weak_order.jl", file, seed=seed))


def generate_weak_order_ext(m: int, seed: int | None = None) -> list[list[list[int]]]:
    weak_order = python_exec(
        run_julia("generate_weak_order_ext.jl", m, WE_file(m), seed=seed)
    )
    return [[int_to_ind_set(i) for i in block] for block in weak_order]
