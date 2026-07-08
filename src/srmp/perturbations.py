import random
from warnings import catch_warnings

import numpy as np
import numpy.typing as npt

with catch_warnings(action="ignore", category=DeprecationWarning):
    from drs import drs  # pyright: ignore[reportMissingTypeStubs]

from src.dataclass import Dataclass, dataclass
from src.random import RNGParam, int_


@dataclass
class PerturbWeight(Dataclass):
    amp: float

    def __call__(self, weights: npt.NDArray[np.float64], rng: RNGParam = None):
        random.seed(int_(rng))

        return np.array(
            drs(
                len(weights),
                1,
                list(np.minimum(weights + self.amp, 1).astype(np.float64)),
                list(np.maximum(weights - self.amp, 0).astype(np.float64)),
            )
        )
