from dataclasses import dataclass, field

from ....dataclass import Dataclass
from ....random import Seed, seeds
from ....random import rng as random_generator


@dataclass
class Seeds(Dataclass):
    A_tr: list[Seed] = field(default_factory=list)
    Mo: list[Seed] = field(default_factory=list)
    Mi: list[Seed] = field(default_factory=list)
    D: list[Seed] = field(default_factory=list)
    Mc: list[Seed] = field(default_factory=list)
    P: list[Seed] = field(default_factory=list)

    @classmethod
    def from_seed(
        cls,
        nb_Atr: int,
        nb_Mo: int,
        nb_Mi: int,
        nb_D: int,
        nb_Mc: int,
        nb_P: int,
        seed: Seed | None = None,
    ):
        rng = random_generator(seed)

        return cls(
            seeds(rng, nb_Atr),
            seeds(rng, nb_Mo),
            seeds(rng, nb_Mi),
            seeds(rng, nb_D),
            seeds(rng, nb_Mc),
            seeds(rng, nb_P),
        )
