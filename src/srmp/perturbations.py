import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from drs import drs
from numpy.random import Generator

from ..dataclass import Dataclass
from ..random import seed


@dataclass
class PerturbWeight(Dataclass):
    amp: float

    def __call__(self, weights: npt.NDArray[np.float64], rng: Generator):
        random.seed(seed(rng))

        return np.array(
            drs(
                len(weights),
                1,
                cast(Sequence[float], np.minimum(weights + self.amp, 1)),
                cast(Sequence[float], np.maximum(weights - self.amp, 0)),
            )
        )
