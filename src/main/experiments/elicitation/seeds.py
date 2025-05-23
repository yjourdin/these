from collections.abc import Sequence
from dataclasses import dataclass, field

from ....dataclass import Dataclass
from ....random import SeedLike, rng_, seeds


@dataclass
class Seeds(Dataclass):
    A_tr: Sequence[SeedLike] = field(default_factory=list)
    A_te: Sequence[SeedLike] = field(default_factory=list)
    Mo: Sequence[SeedLike] = field(default_factory=list)
    D: Sequence[SeedLike] = field(default_factory=list)
    Me: Sequence[SeedLike] = field(default_factory=list)

    @classmethod
    def from_seed(
        cls,
        nb_Atr: int,
        nb_Ate: int,
        nb_Mo: int,
        nb_D: int,
        nb_Me: int,
        seed: SeedLike | None = None,
    ):
        rng = rng_(seed)

        return cls(
            seeds(rng, nb_Atr),
            seeds(rng, nb_Ate),
            seeds(rng, nb_Mo),
            seeds(rng, nb_D),
            seeds(rng, nb_Me),
        )
