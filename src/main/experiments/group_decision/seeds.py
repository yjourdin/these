from collections.abc import Sequence
from dataclasses import dataclass, field

from ....dataclass import Dataclass
from ....random import SeedLike, rng_, seeds


@dataclass
class Seeds(Dataclass):
    A_tr: Sequence[SeedLike] = field(default_factory=list)
    Mo: Sequence[SeedLike] = field(default_factory=list)
    Mi: Sequence[SeedLike] = field(default_factory=list)
    D: Sequence[SeedLike] = field(default_factory=list)
    Mie: Sequence[SeedLike] = field(default_factory=list)
    Mc: Sequence[SeedLike] = field(default_factory=list)
    P: Sequence[SeedLike] = field(default_factory=list)

    @classmethod
    def from_seed(
        cls,
        nb_Atr: int,
        nb_Mo: int,
        nb_Mi: int,
        nb_D: int,
        nb_Mie: int,
        nb_Mc: int,
        nb_P: int,
        seed: SeedLike | None = None,
    ):
        rng = rng_(seed)

        return cls(
            seeds(rng, nb_Atr),
            seeds(rng, nb_Mo),
            seeds(rng, nb_Mi),
            seeds(rng, nb_D),
            seeds(rng, nb_Mie),
            seeds(rng, nb_Mc),
            seeds(rng, nb_P),
        )
