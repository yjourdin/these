from numpy import sort
from numpy.random import Generator, SeedSequence, default_rng
from pandas import DataFrame

from mcda_local.core.performance_table import NormalPerformanceTable, PerformanceTable
from mcda_local.core.power_set import PowerSet
from mcda_local.core.ranker import Ranker
from mcda_local.core.relations import PreferenceStructure
from mcda_local.ranker.rmp import RMP
from mcda_local.ranker.srmp import SRMP


def random_generator(seed=None):
    ss = SeedSequence(seed)
    seed = ss.entropy
    return default_rng(ss), seed


def random_alternatives(nb_alt: int, nb_crit: int, rng: Generator) -> PerformanceTable:
    return NormalPerformanceTable(rng.random((nb_alt, nb_crit)))


def random_srmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable | None = None,
):
    if profiles_values:
        idx = sort(rng.choice(len(profiles_values.data), (nb_profiles, nb_crit)), 0)
        profiles = PerformanceTable(
            DataFrame(
                [
                    profiles_values.data.iloc[idx[:, i], i].to_numpy()
                    for i in range(nb_crit)
                ]
            ).transpose(),
            profiles_values.scales,
        )
    else:
        profiles = NormalPerformanceTable(sort(rng.random((nb_profiles, nb_crit)), 0))
    weights = dict(enumerate(rng.random(nb_crit)))
    lex_order = rng.permutation(nb_profiles)
    return SRMP(weights, profiles, lex_order.tolist())


def random_rmp(
    nb_profiles: int,
    nb_crit: int,
    rng: Generator,
    profiles_values: PerformanceTable | None = None,
) -> RMP:
    if profiles_values:
        idx = sort(rng.choice(len(profiles_values.data), (nb_profiles, nb_crit)))
        profiles = PerformanceTable(
            DataFrame(
                {
                    i: profiles_values.data.iloc[idx[:, i], i].to_list()
                    for i in range(nb_crit)
                }
            ),
            profiles_values.scales,
        )
    else:
        profiles = NormalPerformanceTable(sort(rng.random((nb_profiles, nb_crit)), 0))
    capacities = PowerSet(list(range(nb_crit)))
    lex_order = rng.permutation(nb_profiles)
    return RMP(capacities, profiles, lex_order.tolist())


def random_comparisons(alt: PerformanceTable, model: Ranker) -> PreferenceStructure:
    return model.rank(alt).preference_structure
