from dataclasses import dataclass, field

from ..dataclass import Dataclass
from ..default_max_jobs import DEFAULT_MAX_JOBS
from ..random import Seed
from ..random import rng as random_generator
from ..random import seeds
from .config import create_config
from .directory import RESULTS_DIR
from .fields import GroupConfigField, GroupMeField, GroupMethodField, GroupMoField
from .seeds import Seeds


@dataclass
class Arguments(
    Dataclass, GroupMethodField, GroupMoField, GroupMeField, GroupConfigField
):
    dir: str = RESULTS_DIR
    name: str = ""
    jobs: int = DEFAULT_MAX_JOBS
    stop_error: bool = False
    extend: bool = False
    seed: Seed | None = None
    nb_A_tr: int = 1
    nb_Mo: int | None = None
    nb_A_te: int | None = None
    nb_D: int | None = None
    nb_Me: int | None = None
    seeds: Seeds = field(default_factory=Seeds)
    N_tr: list[int] = field(default_factory=list)
    N_te: list[int] | None = None
    group_size: list[int] = field(default_factory=lambda: [1])
    M: list[int] = field(default_factory=list)
    Ko: list[int] = field(default_factory=list)
    N_bc: list[int] = field(default_factory=list)
    same_alt: list[bool] = field(default_factory=lambda: [True])
    Ke: list[int] | None = None
    error: list[float] = field(default_factory=lambda: [0])

    def complete(self):
        # Create random seeds
        rng = random_generator(self.seed)

        nb_A_tr = self.nb_A_tr
        nb_Mo = self.nb_Mo or nb_A_tr
        nb_D = self.nb_D or nb_Mo
        nb_Me = self.nb_Me or nb_D
        nb_A_te = self.nb_A_te or nb_Me

        self.seeds.A_tr += seeds(rng, nb_A_tr - len(self.seeds.A_tr))
        self.seeds.A_te = self.seeds.A_te + seeds(rng, nb_A_te - len(self.seeds.A_te))
        for group_size in self.group_size:
            self.seeds.Mo[group_size] = self.seeds.Mo.get(group_size, []) + seeds(
                rng, nb_Mo - len(self.seeds.Mo.get(group_size, []))
            )
        self.seeds.D = self.seeds.D + seeds(rng, nb_D - len(self.seeds.D))
        self.seeds.Me = self.seeds.Me + seeds(rng, nb_Me - len(self.seeds.Me))

        # Create missing configs
        for method in self.method:
            if not any(config.method is method for config in self.config):
                self.config.append(create_config(method=method))
