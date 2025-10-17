import ast
from itertools import chain
from subprocess import run
from typing import Any


def run_julia(scriptname: str, *args: Any, **kwargs: Any):
    return run(
        [scriptname]
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
    s = s.replace("Int64", "")
    try:
        return ast.literal_eval(s)
    except Exception:
        raise Exception(f"Julia output : {s}")


def generate_linext(m: int, seed: int | None = None) -> list[list[int]]:
    return python_exec(run_julia("generate_linext", m, seed=seed))


def generate_weak_order(m: int, seed: int | None = None) -> list[int]:
    return python_exec(run_julia("generate_weak_order", m, seed=seed))


def generate_weak_order_ext(m: int, seed: int | None = None) -> list[list[list[int]]]:
    return python_exec(run_julia("generate_weak_order_ext", m, seed=seed))
