import ast
from itertools import chain
from subprocess import run
from typing import Any

from ..random import Seed
from .file import PARENT_DIR, S_file, WE_file


def run_julia(scriptname: str, *args: Any, **kwargs: Any):
    return run(
        ["julia", PARENT_DIR / scriptname]
        + [str(x) for x in args]
        + list(
            chain.from_iterable(
                (f"--{k}", str(v)) for k, v in kwargs.items() if v is not None
            )
        ),
        capture_output=True,
        text=True,
    ).stdout


def generate_linext(m: int, seed: Seed | None = None) -> list[list[bool]]:
    linext = ast.literal_eval(run_julia("generate_linext.jl", m, seed=seed))
    return [[bool(int(x)) for x in node] for node in linext]


def generate_partial_sum(m: int) -> None:
    return ast.literal_eval(run_julia("generate_partial_sum.jl", m, output=S_file(m)))


def generate_weak_order(m: int, seed: Seed | None = None) -> list[int]:
    file = S_file(m)

    if not file.exists():
        generate_partial_sum(m)

    return ast.literal_eval(run_julia("generate_weak_order.jl", m, file, seed=seed))


def generate_weak_order_ext(m: int, seed: Seed | None = None) -> list[list[list[bool]]]:
    weak_order = ast.literal_eval(
        run_julia("generate_weak_order_ext.jl", m, WE_file(m), seed=seed)
    )
    return [[[bool(int(x)) for x in node] for node in block] for block in weak_order]
