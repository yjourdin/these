from typing import Any, Literal, cast

from scipy.stats import kendalltau

from argument_parser import parse_args
from generate_random import (
    random_alternatives,
    random_comparisons,
    random_generator,
    random_rmp,
    random_srmp,
)
from mcda_local.core.relations import PreferenceStructure
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
args = parse_args(["@config.txt"])


# Cast arguments
Model = Literal["RMP", "SRMP"]
Method = Literal["MIP", "SA"]

n_tr: int = args.n_tr
n_te: int = args.n_te
m: int = args.m
k_o: int = args.k_o
k_e: int = args.k_e
n_bc: int = args.n_bc
method: Method = args.method
model: Model = args.model or "SRMP"
A_train_seed = cast(int, args.A_train_seed)
model_seed = cast(int, args.model_seed)
D_train_seed = cast(int, args.D_train_seed)
learn_seed = cast(int, args.learn_seed)
A_test_seed = cast(int, args.A_test_seed)


# Create random generators
A_train_rng, A_train_seed = random_generator(A_train_seed)
model_rng, model_seed = random_generator(model_seed)
D_train_rng, D_train_seed = random_generator(D_train_seed)
learn_rng, learn_seed = random_generator(learn_seed)
A_test_rng, A_test_seed = random_generator(A_test_seed)


# Generate training data set of alternatives
A_train = random_alternatives(n_tr, m, A_train_rng)


# Generate original model
match model:
    case "RMP":
        Mo = random_rmp(k_o, m, model_rng)
    case "SRMP":
        Mo = random_srmp(k_o, m, model_rng)


# Generate training binary comparisons
bc_train = random_comparisons(A_train, Mo).relations
train_index = D_train_rng.choice(len(bc_train), n_bc, replace=False)
D_train = PreferenceStructure([bc_train[i] for i in train_index])


# Create learner
learner: MIP | SimulatedAnnealing[RMP | SRMP]
kwargs: dict[Any, Any] = {}
match method:
    case "MIP":
        learner = MIP(
            gamma=args.gamma,
            non_dictator=args.non_dictator,
        )

        kwargs["lexicographic_order"] = args.lexicographic_order
        kwargs["profiles_number"] = args.profiles_number
        kwargs["max_profiles_number"] = args.max_profiles_number

    case "SA":
        neighbors: list[Neighbor] = []
        neighbors.append(NeighborProfiles(midpoints(A_train)))
        match model:
            case "RMP":
                neighbors.append(NeighborCapacities())
            case "SRMP":
                neighbors.append(NeighborWeights(0.01))
        if k_e >= 2:
            neighbors.append(NeighborLexOrder())

        learner = SimulatedAnnealing(
            args.T0, args.Tf, args.alpha, args.L, RandomNeighbor(neighbors)
        )

        match model:
            case "RMP":
                kwargs["initial_model"] = random_rmp(
                    k_e, m, learn_rng, midpoints(A_train)
                )
            case "SRMP":
                kwargs["initial_model"] = random_srmp(k_e, m, learn_rng)
        kwargs["rng"] = learn_rng


# Learn
Me = learner.learn(A_train, D_train, **kwargs)


# Compare results
if not Me:
    ValueError("No elicited model")
else:
    A_test = random_alternatives(n_te, m, A_test_rng)

    restauration = Me.fitness(A_train, D_train)

    ranking_o = Mo.rank(A_test)
    ranking_e = Me.rank(A_test)

    kendall_tau = kendalltau(ranking_o.data, ranking_e.data)

    print(f"Restauration rate : {restauration}")
    print(f"Kendall's tau : {kendall_tau.statistic}")