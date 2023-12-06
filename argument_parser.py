import argparse
from typing import Sequence


def parse_args(args: Sequence[str] | None = None):
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Learn a preference model.", fromfile_prefix_chars="@"
    )

    # Constant numbers
    parser.add_argument(
        "--n-tr", type=int, required=True, help="Number of training alternatives"
    )
    parser.add_argument(
        "--n-te", type=int, required=True, help="Number of testing alternatives"
    )
    parser.add_argument("--m", type=int, required=True, help="Number of criteria")
    parser.add_argument(
        "--k-o",
        type=int,
        required=True,
        help="Number of profiles in the original model",
    )
    parser.add_argument(
        "--k-e",
        type=int,
        required=True,
        help="Number of profiles in the elicited model",
    )
    parser.add_argument(
        "--n-bc", type=int, required=True, help="Number of training binary comparisons"
    )

    # Method used
    subparsers = parser.add_subparsers(
        help="Method used to learn", dest="method", required=True
    )

    # Mixed Integer Program arguments
    parser_mip = subparsers.add_parser("MIP", help="Mixed-Integer Program")
    given_group = parser_mip.add_mutually_exclusive_group(required=True)
    given_group.add_argument(
        "--lexicographic-order", type=int, help="Lexicographic order"
    )
    given_group.add_argument("--profiles-number", type=int, help="Number of profiles")
    given_group.add_argument(
        "--max-profiles-number", type=int, help="Maximum number of profiles"
    )
    parser_mip.add_argument(
        "--gamma",
        default=0.001,
        type=float,
        help="Value used for modeling strict inequalities",
    )
    parser_mip.add_argument(
        "--non-dictator",
        action="store_true",
        default=False,
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
    parser_sa.add_argument("--Tf", type=float, required=True, help="Final temperature")
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
        "--time-limit", type=int, help="Time limit (in seconds)"
    )
    stopping_criterion_group.add_argument(
        "--iteration-limit", type=int, help="Max number of iterations"
    )
    stopping_criterion_group.add_argument(
        "--non-improving-limit",
        type=int,
        help="Max number of non improving iterations",
    )

    # Seed arguments
    seeds_group = parser.add_argument_group(
        "seeds", "Random seeds used for reproductibility"
    )
    seeds_group.add_argument(
        "--experiment-seed",
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
    return parser.parse_args(args)
