from dataclasses import asdict, dataclass, field
from json import dumps, loads

from numpy.random import default_rng

from ..jobs import JOBS
from .config import Config, create_config
from .directory import RESULTS_DIR
from .seed import Seeds, seeds
from .type import Method, Model


def config_hook(dct: dict):
    try:
        return create_config(**dct)
    except TypeError:
        return dct


@dataclass
class Arguments:
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
    N_tr: list[int] = field(default_factory=list)
    N_te: list[int] = field(default_factory=list)
    method: list[Method] = field(default_factory=list)
    M: list[int] = field(default_factory=list)
    Mo: list[Model] = field(default_factory=list)
    Ko: list[int] = field(default_factory=list)
    N_bc: list[int] = field(default_factory=list)
    Me: list[Model] = field(default_factory=list)
    Ke: list[int] = field(default_factory=list)
    error: list[float] = field(default_factory=list)
    config: list[Config] = field(default_factory=list)

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    @classmethod
    def from_json(cls, s):
        return cls.from_dict(loads(s, object_hook=config_hook))

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return dumps(self.to_dict(), indent=4)

    def complete(self):
        # Create random seeds
        rng = default_rng(self.seed)

        self.seeds.A_train = self.seeds.A_train + seeds(
            rng, self.nb_A_tr - len(self.seeds.A_train)
        )
        self.seeds.Mo = self.seeds.Mo + seeds(
            rng, (self.nb_Mo or self.nb_A_tr) - len(self.seeds.Mo)
        )
        self.seeds.A_test = self.seeds.A_test + seeds(
            rng, (self.nb_A_te or self.nb_Mo or self.nb_A_tr) - len(self.seeds.A_test)
        )
        self.seeds.D = self.seeds.D + seeds(
            rng,
            (self.nb_D or self.nb_A_te or self.nb_Mo or self.nb_A_tr)
            - len(self.seeds.D),
        )
        self.seeds.Me = self.seeds.Me + seeds(
            rng,
            (self.nb_Me or self.nb_D or self.nb_A_te or self.nb_Mo or self.nb_A_tr)
            - len(self.seeds.Me),
        )

        # Create missing configs
        for method in self.method:
            if not any(config.method == method for config in self.config):
                self.config.append(create_config(method=method))
