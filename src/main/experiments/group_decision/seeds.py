from dataclasses import dataclass, field

from ....dataclass import Dataclass
from ....random import Seed, seeds
from ....random import rng as random_generator


@dataclass
class Seeds(Dataclass):
    A_tr: list[Seed] = field(default_factory=list)
    Mo: list[Seed] = field(default_factory=list)
    D: list[Seed] = field(default_factory=list)

    @classmethod
    def from_seed(
        cls,
        nb_Atr: int,
        nb_Ate: int,
        nb_Mo: int,
        nb_D: int,
        nb_Me: int,
        seed: Seed | None = None,
    ):
        rng = random_generator(seed)

        return cls(
            seeds(rng, nb_Atr),
            seeds(rng, nb_Mo),
            seeds(rng, nb_D),
        )
