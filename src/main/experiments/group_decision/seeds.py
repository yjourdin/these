from collections.abc import Sequence

from src.dataclass import Dataclass, dataclass, field
from src.random import SeedLike, int_, rng_


@dataclass
class Seeds(Dataclass):
    A_tr: Sequence[int] = field(default_factory=list)
    Mo: Sequence[int] = field(default_factory=list)
    Mi: Sequence[int] = field(default_factory=list)
    D: Sequence[int] = field(default_factory=list)
    Mie: Sequence[int] = field(default_factory=list)
    Mc: Sequence[int] = field(default_factory=list)
    P: Sequence[int] = field(default_factory=list)

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
            [int_(s) for s in rng.spawn(nb_Atr)],
            [int_(s) for s in rng.spawn(nb_Mo)],
            [int_(s) for s in rng.spawn(nb_Mi)],
            [int_(s) for s in rng.spawn(nb_D)],
            [int_(s) for s in rng.spawn(nb_Mie)],
            [int_(s) for s in rng.spawn(nb_Mc)],
            [int_(s) for s in rng.spawn(nb_P)],
        )
