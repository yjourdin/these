import argparse
from dataclasses import dataclass
from types import NoneType, UnionType
from typing import Any, Literal, Sequence, get_args, get_origin

Model = Literal["RMP", "SRMP"]
Method = Literal["MIP", "SA"]


@dataclass(frozen=True)
class Arguments:
    N_tr: int
    N_te: int
    M: int
    K_o: int
    K_e: int
    N_bc: int

    method: Method
    model: Model = "SRMP"

    gamma: float | None = None
    non_dictator: bool | None = None
    lexicographic_order: list[int] | None = None
    profiles_number: int | None = None
    max_profiles_number: int | None = None

    T0: float | None = None
    alpha: float | None = None
    L: int | None = None
    Tf: float | None = None
    max_time: int | None = None
    max_iter: int | None = None
    max_iter_non_improving: int | None = None

    seed: int | None = None
    A_train_seed: int | None = None
    model_seed: int | None = None
    D_train_seed: int | None = None
    learn_seed: int | None = None
    A_test_seed: int | None = None

    def __post_init__(self):
        for name, field_type in self.__annotations__.items():
            if (not get_origin(field_type) is UnionType) or (
                NoneType not in get_args(field_type)
            ):
                if get_origin(field_type) is Literal:
                    if self.__dict__[name] not in get_args(field_type):
                        current_type = type(self.__dict__[name])
                        raise TypeError(
                            f"The field `{name}` "
                            f"was assigned by `{current_type}` "
                            f"instead of `{field_type}`"
                        )
                else:
                    if not isinstance(self.__dict__[name], field_type):
                        current_type = type(self.__dict__[name])
                        raise TypeError(
                            f"The field `{name}` "
                            f"was assigned by `{current_type}` "
                            f"instead of `{field_type}`"
                        )

    def kwargs(
        self, args: Sequence[str], keys: list[str] | dict[str, str] | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for i, arg in enumerate(args):
            arg_value = getattr(self, arg)
            if arg_value:
                match keys:
                    case list():
                        key = keys[i]
                    case dict():
                        key = keys.get(arg, arg)
                    case _:
                        key = arg
                result[key] = arg_value
        return result


def parse_args(args: Sequence[str] | None = None):
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Learn a preference model.", fromfile_prefix_chars="@"
    )

    # Constant numbers
    parser.add_argument("--N-tr", type=int, help="Number of training alternatives")
    parser.add_argument("--N-te", type=int, help="Number of testing alternatives")
    parser.add_argument("--M", type=int, help="Number of criteria")
    parser.add_argument(
        "--K-o",
        type=int,
        help="Number of profiles in the original model",
    )
    parser.add_argument(
        "--K-e",
        type=int,
        help="Number of profiles in the elicited model",
    )
    parser.add_argument(
        "--N-bc", type=int, help="Number of training binary comparisons"
    )

    # Method used
    subparsers = parser.add_subparsers(help="Method used to learn", dest="method")

    # Mixed Integer Program arguments
    parser_mip = subparsers.add_parser("MIP", help="Mixed-Integer Program")
    given_group = parser_mip.add_mutually_exclusive_group(required=True)
    given_group.add_argument(
        "--lexicographic-order", nargs="+", type=int, help="Lexicographic order"
    )
    given_group.add_argument("--profiles-number", type=int, help="Number of profiles")
    given_group.add_argument(
        "--max-profiles-number", type=int, help="Maximum number of profiles"
    )
    parser_mip.add_argument(
        "--gamma",
        type=float,
        help="Value used for modeling strict inequalities",
    )
    parser_mip.add_argument(
        "--non-dictator",
        action="store_true",
        help="Prevent dictator weights",
    )

    # Simulated Annealing arguments
    parser_sa = subparsers.add_parser("SA", help="Simulated Annealing")
    parser_sa.add_argument(
        "--model", choices=["RMP", "SRMP"], required=True, help="The model to learn"
    )
    parser_sa.add_argument(
        "--T0", type=float, required=True, help="Initial temperature"
    )
    parser_sa.add_argument(
        "--alpha", type=float, required=True, help="Temperature decrease"
    )
    parser_sa.add_argument(
        "--L",
        type=int,
        required=True,
        help="Number of iterations for each temperature step",
    )
    stopping_criterion_group = parser_sa.add_mutually_exclusive_group(required=True)
    stopping_criterion_group.add_argument("--Tf", type=float, help="Final temperature")
    stopping_criterion_group.add_argument(
        "--max-time", type=int, help="Time limit (in seconds)"
    )
    stopping_criterion_group.add_argument(
        "--max-iter", type=int, help="Max number of iterations"
    )
    stopping_criterion_group.add_argument(
        "--max-iter-non-improving",
        type=int,
        help="Max number of non improving iterations",
    )

    # Seed arguments
    seeds_group = parser.add_argument_group(
        "seeds", "Random seeds used for reproductibility"
    )
    seeds_group.add_argument(
        "--seed",
        type=int,
        help="Seed for all the experiment",
    )
    seeds_group.add_argument(
        "--A-train-seed",
        type=int,
        help="Seed used to generate the training data set of alternatives",
    )
    seeds_group.add_argument(
        "--model-seed", type=int, help="Seed used to generate the ground truth model"
    )
    seeds_group.add_argument(
        "--D-train-seed",
        type=int,
        help="Seed used to select the alternatives into the training dataset",
    )
    seeds_group.add_argument(
        "--learn-seed", type=int, help="Seed used during the learning process"
    )
    seeds_group.add_argument(
        "--A-test-seed",
        type=int,
        help="Seed used to generate the test data set of alternatives",
    )
    # parser.add_argument("--inconsistencies",
    # action="store_true", help="Take into account inconsistent comparisons")

    # Parse arguments
    pua = None
    ns, ua = parser.parse_known_args(args)
    while len(ua) and ua != pua:
        ns, ua = parser.parse_known_args(ua, ns)
        pua = ua
    ns = parser.parse_args(ua, ns)
    return Arguments(**vars(ns))
