from dataclasses import dataclass, field

from numpy.random import default_rng

from ..dataclass import Dataclass
from ..jobs import JOBS
from ..methods import MethodEnum
from ..models import GroupModelEnum
from ..seed import seed, seeds
from .config import Config, create_config
from .directory import RESULTS_DIR
from .seeds import Seeds, group_seed


@dataclass
class Arguments(Dataclass):
    dir: str = RESULTS_DIR
    name: str = ""
    jobs: int = JOBS
    seed: int | None = None
    nb_A_tr: int = 1
    nb_Mo: int | None = None
    nb_A_te: int | None = None
    nb_D: int | None = None
    nb_Me: int | None = None
    seeds: Seeds = Seeds()
    group_seeds: dict[int, list[int]] = field(default_factory=dict)
    N_tr: list[int] = field(default_factory=list)
    N_te: list[int] = field(default_factory=list)
    group_size: list[int] = [1]
    method: list[MethodEnum] = field(default_factory=list)
    M: list[int] = field(default_factory=list)
    Mo: list[GroupModelEnum] = field(default_factory=list)
    Ko: list[int] = field(default_factory=list)
    N_bc: list[int] = field(default_factory=list)
    Me: list[GroupModelEnum] = field(default_factory=list)
    Ke: list[int] = field(default_factory=list)
    error: list[float] = field(default_factory=list)
    config: list[Config] = field(default_factory=list)

    def complete(self):
        # Create random seeds
        rng = default_rng(self.seed)

        nb_A_tr = self.nb_A_tr
        nb_Mo = self.nb_Mo or nb_A_tr
        nb_D = self.nb_D or nb_Mo
        nb_Me = self.nb_Me or nb_D
        nb_A_te = self.nb_A_te or nb_Me

        self.seeds.A_tr += seeds(rng, nb_A_tr - len(self.seeds.A_tr))
        self.seeds.A_te = self.seeds.A_te + seeds(rng, nb_A_te - len(self.seeds.A_te))
        for size in self.group_size:
            for group_id in range(len(self.seeds.Mo[size])):
                if self.seeds.Mo[size][group_id].group == -1:
                    self.seeds.Mo[size][group_id].group = seed(rng)
                self.seeds.Mo[size][group_id].dm += seeds(
                    default_rng(self.seeds.Mo[size][group_id].group),
                    size - len(self.seeds.Mo[size][group_id].dm),
                )
            self.seeds.Mo[size] += [
                group_seed(seed, size)
                for seed in seeds(rng, nb_Mo - len(self.seeds.Mo[size]))
            ]
        self.seeds.D = self.seeds.D + seeds(rng, nb_D - len(self.seeds.D))
        self.seeds.Me = self.seeds.Me + seeds(rng, nb_Me - len(self.seeds.Me))

        # Create missing configs
        for method in self.method:
            if not any(config.method == method for config in self.config):
                self.config.append(create_config(method=method))
