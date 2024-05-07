from typing import cast

from numpy.random import default_rng

from .arguments import Arguments


def create_seeds(args: Arguments):
    seeds = {}
    rng = default_rng(args.seed)

    seeds["A_train"] = (
        args.A_tr_seeds
        if isinstance(args.A_tr_seeds, list)
        else cast(list[int], rng.integers(2**63, size=args.A_tr_seeds).tolist())
    )

    seeds["A_test"] = (
        args.A_te_seeds
        if isinstance(args.A_te_seeds, list)
        else cast(list[int], rng.integers(2**63, size=args.A_te_seeds).tolist())
    )

    seeds["Mo"] = (
        args.Mo_seeds
        if isinstance(args.Mo_seeds, list)
        else cast(list[int], rng.integers(2**63, size=args.Mo_seeds).tolist())
    )

    return seeds
