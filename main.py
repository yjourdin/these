from math import exp
from typing import Any
from numpy import sort
from numpy.random import default_rng, SeedSequence
from mcda.core.performance_table import PerformanceTable
from mcda.core.scales import QuantitativeScale
from mcda.outranking.srmp import SRMP
from mcda.core.interfaces import Learner
from mcda.core.relations import PreferenceStructure


def generate_alternatives(nb_alt, nb_crit, rng):
    return PerformanceTable(
        rng.random((nb_alt, nb_crit)),
        dict().fromkeys(range(nb_crit), QuantitativeScale(0, 1)))


def generate_model(nb_profiles, nb_crit, rng):
    profiles = PerformanceTable(
        sort(rng.random((nb_profiles, nb_crit)), 0),
        dict().fromkeys(range(nb_crit), QuantitativeScale(0, 1)))
    weights = dict(enumerate(rng.random(nb_crit)))
    lex_order = rng.permutation(nb_profiles)
    return SRMP(weights, profiles, lex_order.tolist())


def generate_comparisons(alt, model):
    return model.rank(alt).preference_structure


def copy(model):
    return SRMP(model.criteria_weights, model.profiles, model.lexicographic_order)


def fitness(model, performance_table, comparisons):
    rank = model.rank(performance_table)
    return sum(rel in rank.preference_structure.transitive_closure for rel in comparisons) / len(comparisons.relations)


class Neighbor:
    def __init__(self):
        pass

    def __call__(self, model, rng):
        return copy(model)


class NeighborProfiles(Neighbor):
    def __init__(self, amp):
        self.amp = amp

    def __call__(self, model, rng):
        neighbor = copy(model)
        df = neighbor.profiles.data
        index = (rng.integers(0, df.shape[0]), rng.integers(0, df.shape[1]))
        x = rng.uniform(-1, 1)
        neighbor.profiles.data.iat[index] = min(
            max(df.iat[index] + self.amp * x, 0), 1)
        return neighbor


class NeighborWeights(Neighbor):
    def __init__(self, amp):
        self.amp = amp

    def __call__(self, model, rng):
        neighbor = copy(model)
        d = neighbor.criteria_weights
        keys = list(d)
        crit = keys[rng.integers(0, len(keys))]
        x = rng.uniform(-1, 1)
        neighbor.criteria_weights[crit] = min(
            max(d[crit] + self.amp * x, 0), 1)
        return neighbor


class NeighborLexOrder(Neighbor):
    def __init__(self):
        super().__init__()

    def __call__(self, model, rng):
        neighbor = copy(model)
        lex_order = neighbor.lexicographic_order
        i = rng.integers(0, len(lex_order) - 1)
        neighbor.lexicographic_order[i], neighbor.lexicographic_order[i + 1] = (
            lex_order[i+1], lex_order[i])
        return neighbor


class RandomNeighbor(Neighbor):
    def __init__(self, profiles_amp, weighs_amp):
        self.neighbors = [NeighborProfiles(profiles_amp), NeighborWeights(
            weighs_amp), NeighborLexOrder()]

    def __call__(self, model, rng):
        if len(model.lexicographic_order) >= 2:
            i = rng.integers(0, len(self.neighbors))
        else:
            i = rng.integers(0, len(self.neighbors)-1)
        neighbor = self.neighbors[i]
        return self.neighbors[i](model, rng)


class SimulatedAnnealing():
    def __init__(self, f, T0, Tmax,  alpha, L, neighbor):
        self.f = f
        self.T0 = T0
        self.Tmax = Tmax
        self.alpha = alpha
        self.L = L
        self.neighbor = neighbor

    def optimise(self, s0, rng):
        T = self.T0
        s_star = s0
        f_star = self.f(s_star)
        while T > self.Tmax:
            for i in range(self.L):
                s_prime = self.neighbor(s_star, rng)
                f_prime = self.f(s_prime)
                if rng.random() < exp((f_star - f_prime) / T):
                    s_star = s_prime
                    f_star = f_prime
            T = self.alpha * T
            print(f"Temperature : {T}")
        return s_star


class SRMPLearnerSimulatedAnnealing(Learner):

    def __init__(
            self,
            performance_table,
            comparisons,
            nb_profiles):
        self.performance_table = performance_table
        self.comparisons = comparisons
        self.nb_profiles = nb_profiles

    def learn(self, rng):
        initial = generate_model(self.nb_profiles, len(
            self.performance_table.criteria), rng)
        sa = SimulatedAnnealing(
            lambda model: 1 -
            fitness(model, self.performance_table, self.comparisons),
            0.01,
            0.005,
            0.995,
            1,
            RandomNeighbor(0.1, 0.1))
        return sa.optimise(initial, rng)


def generate_random_generator(seed=None):
    ss = SeedSequence(seed)
    seed = ss.entropy
    return default_rng(ss), seed


def test(
        nb_alt,
        nb_crit,
        nb_profiles,
        D_train_size,
        A_train_seed=None,
        model_seed=None,
        D_train_seed=None,
        learn_seed=None,
        A_test_seed=None):
    A_train_rng, A_train_seed = generate_random_generator(A_train_seed)
    model_rng, model_seed = generate_random_generator(model_seed)
    D_train_rng, D_train_seed = generate_random_generator(D_train_seed)
    learn_rng, learn_seed = generate_random_generator(learn_seed)
    A_test_rng, A_test_seed = generate_random_generator(A_test_seed)

    A_train = generate_alternatives(nb_alt, nb_crit, A_train_rng)

    Mo = generate_model(nb_profiles, nb_crit, model_rng)

    comparisons_train = generate_comparisons(A_train, Mo).relations
    train_index = D_train_rng.integers(
        0,
        len(comparisons_train),
        D_train_size)
    D_train = PreferenceStructure([comparisons_train[i] for i in train_index])

    learner = SRMPLearnerSimulatedAnnealing(A_train, D_train, nb_profiles)
    Me = learner.learn(learn_rng)

    A_test = generate_alternatives(nb_alt, nb_crit, A_test_rng)

    Ranking_o = Mo.rank(A_test)
    Ranking_e = Me.rank(A_test)

    return Mo, Me, Ranking_o, Ranking_e, A_train, A_test


Mo, Me, Ro, Re, atrain, atest = test(10, 3, 1, 10)
