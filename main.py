from dataclasses import fields
from time import time
from typing import Any, cast

from numpy.random import SeedSequence, default_rng
from scipy.stats import kendalltau

from argument_parser import parse_args
from generate import (
    all_comparisons,
    balanced_rmp,
    balanced_srmp,
    random_alternatives,
    random_comparisons,
    random_rmp,
    random_srmp,
)
from mcda_local.core.learner import Learner
from mcda_local.learner.mip import MIP
from mcda_local.learner.neighbor import Neighbor, RandomNeighbor
from mcda_local.learner.sa import SimulatedAnnealing
from mcda_local.ranker.rmp import RMP
from mcda_local.ranker.srmp import SRMP
from neighbors import (
    NeighborCapacities,
    NeighborLexOrder,
    NeighborProfiles,
    NeighborWeights,
)
from utils import midpoints

# Parse arguments
ARGS = parse_args()


# Create random seeds
seeds: dict[str, int] = {"general": cast(int, SeedSequence(ARGS.seed).entropy)}
seeds.update(
    zip(
        ["A_train", "model", "D_train", "learn", "A_test"],
        SeedSequence(seeds["general"]).generate_state(5),
    )
)
seeds.update(
    ARGS.kwargs(
        ["A_train_seed", "model_seed", "D_train_seed", "learn_seed", "A_test_seed"],
        ["A_train", "model", "D_train", "learn", "A_test"],
    )
)

# Generate training data set of alternatives
A_train = random_alternatives(ARGS.N_tr, ARGS.M, default_rng(seeds["A_train"]))


# Generate original model
match ARGS.model:
    case "RMP":
        Mo = random_rmp(ARGS.K_o, ARGS.M, default_rng(seeds["model"]))
    case "SRMP":
        Mo = random_srmp(ARGS.K_o, ARGS.M, default_rng(seeds["model"]))
print(Mo)


# Generate training binary comparisons
D_train = random_comparisons(ARGS.N_bc, A_train, Mo, default_rng(seeds["D_train"]))


# Create learner
learner: Learner[RMP | SRMP]
learn_kwargs: dict[str, Any] = {}
match ARGS.method:
    case "MIP":
        # Create MIP leaner
        learner = MIP(
            **ARGS.kwargs(
                [
                    "max_profiles_number",
                    "profiles_number",
                    "lexicographic_order",
                    "gamma",
                    "non_dictator",
                ]
            )
        )

    case "SA":
        # Create neighbors
        neighbors: list[Neighbor] = []
        neighbors.append(NeighborProfiles(midpoints(A_train)))
        match ARGS.model:
            case "RMP":
                neighbors.append(NeighborCapacities())
            case "SRMP":
                neighbors.append(NeighborWeights(0.1))
        if ARGS.K_e >= 2:
            neighbors.append(NeighborLexOrder())

        # Create SA leaner
        learner = SimulatedAnnealing(
            neighbor=RandomNeighbor(neighbors),
            **ARGS.kwargs(
                [
                    "T0",
                    "alpha",
                    "L",
                    "Tf",
                    "max_time",
                    "max_iter",
                    "max_iter_non_improving",
                ]
            ),
        )

        # Add learning kwargs
        seeds.update(
            zip(
                ["initial_model", "sa"],
                SeedSequence(seeds["learn"]).generate_state(2),
            )
        )
        match ARGS.model:
            case "RMP":
                learn_kwargs["initial_model"] = balanced_rmp(
                    ARGS.K_e,
                    ARGS.M,
                    default_rng(seeds["initial_model"]),
                    midpoints(A_train),
                )
            case "SRMP":
                learn_kwargs["initial_model"] = balanced_srmp(
                    ARGS.K_e,
                    ARGS.M,
                    default_rng(seeds["initial_model"]),
                    midpoints(A_train),
                )
        learn_kwargs["rng"] = default_rng(seeds["sa"])


# Learn
# print("learning")
learning_start_time = time()
Me = learner.learn(A_train, D_train, **learn_kwargs)
learning_end_time = time()
learning_total_time = learning_end_time - learning_start_time
# print("learned")


# Compare results
if not Me:
    ValueError("No elicited model")
else:
    A_test = random_alternatives(ARGS.N_te, ARGS.M, default_rng(seeds["A_test"]))

    train_accuracy = Me.fitness(A_train, D_train)
    test_accuracy = Me.fitness(A_test, all_comparisons(A_test, Mo))

    ranking_o = Mo.rank(A_test)
    ranking_e = Me.rank(A_test)

    kendall_tau = kendalltau(ranking_o.data, ranking_e.data)

    # Print results
    result: str = ""
    result += str(learning_total_time)
    result += "," + str(train_accuracy)
    result += "," + str(test_accuracy)
    result += "," + str(kendall_tau.statistic)
    for seed in seeds.values():
        result += "," + str(seed)
    for arg in fields(ARGS):
        result += "," + str(getattr(ARGS, arg.name))

    print(result)
