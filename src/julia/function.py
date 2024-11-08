from itertools import chain
from pathlib import Path
from subprocess import run

from ..random import Seed
from .file import S_file, WE_dir


def run_julia(scriptname: str, *args, **kwargs):
    return run(
        ["julia", Path("src/julia") / scriptname]
        + [str(x) for x in args]
        + list(chain.from_iterable((f"--{k}", str(v)) for k, v in kwargs.items())),
        capture_output=True,
        text=True,
    ).stdout


def generate_linext(m: int, seed: Seed | None = None):
    return run_julia("generate_linext.jl", m, seed=seed)


def generate_weak_order(m: int, seed: Seed | None = None):
    return run_julia("generate_weak_order.jl", m, S_file(m), seed=seed)


def generate_weak_order_ext(m: int, seed: Seed | None = None):
    return run_julia("generate_weak_order_ext.jl", m, WE_dir(m), seed=seed)
