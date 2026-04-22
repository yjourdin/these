from collections.abc import Sequence

from src.dataclass import Dataclass, dataclass, field
from src.random import Seed, SeedLike, rng_, seed_


@dataclass
class Seeds(Dataclass):
    A_tr: Sequence[Seed] = field(default_factory=list)
    A_te: Sequence[Seed] = field(default_factory=list)
    Mo: Sequence[Seed] = field(default_factory=list)
    D: Sequence[Seed] = field(default_factory=list)
    Me: Sequence[Seed] = field(default_factory=list)

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
            seed_(rng).spawn(nb_Atr),
            seed_(rng).spawn(nb_Ate),
            seed_(rng).spawn(nb_Mo),
            seed_(rng).spawn(nb_D),
            seed_(rng).spawn(nb_Me),
        )
