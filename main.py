from typing import Any

from scipy.stats import kendalltau

from argument_parser import parse_args
from generate_random import (
    random_alternatives,
    random_comparisons,
    random_generator,
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


# Create random generators
default_rng, default_seed = random_generator(ARGS.seed)

A_train_rng, A_train_seed = (
    random_generator(ARGS.A_train_seed)
    if ARGS.A_train_seed
    else (default_rng, default_seed)
)
model_rng, model_seed = (
    random_generator(ARGS.model_seed)
    if ARGS.model_seed
    else (default_rng, default_seed)
)
D_train_rng, D_train_seed = (
    random_generator(ARGS.D_train_seed)
    if ARGS.D_train_seed
    else (default_rng, default_seed)
)
learn_rng, learn_seed = (
    random_generator(ARGS.learn_seed)
    if ARGS.learn_seed
    else (default_rng, default_seed)
)
A_test_rng, A_test_seed = (
    random_generator(ARGS.A_test_seed)
    if ARGS.A_test_seed
    else (default_rng, default_seed)
)


# Generate training data set of alternatives
A_train = random_alternatives(ARGS.n_tr, ARGS.m, A_train_rng)


# Generate original model
match ARGS.model:
    case "RMP":
        Mo = random_rmp(ARGS.k_o, ARGS.m, model_rng)
    case "SRMP":
        Mo = random_srmp(ARGS.k_o, ARGS.m, model_rng)


# Generate training binary comparisons
D_train = random_comparisons(ARGS.n_bc, A_train, Mo, D_train_rng)


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
                neighbors.append(NeighborWeights(0.01))
        if ARGS.k_e >= 2:
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
        match ARGS.model:
            case "RMP":
                learn_kwargs["initial_model"] = random_rmp(
                    ARGS.k_e, ARGS.m, learn_rng, midpoints(A_train)
                )
            case "SRMP":
                learn_kwargs["initial_model"] = random_srmp(ARGS.k_e, ARGS.m, learn_rng)
        learn_kwargs["rng"] = learn_rng


# Learn
Me = learner.learn(A_train, D_train, **learn_kwargs)


# Compare results
if not Me:
    ValueError("No elicited model")
else:
    A_test = random_alternatives(ARGS.n_te, ARGS.m, A_test_rng)

    restauration = Me.fitness(A_train, D_train)

    ranking_o = Mo.rank(A_test)
    ranking_e = Me.rank(A_test)

    kendall_tau = kendalltau(ranking_o.data, ranking_e.data)

    print(f"Restauration rate : {restauration}")
    print(f"Kendall's tau : {kendall_tau.statistic}")
